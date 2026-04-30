import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import HTTPException, status

from ..config import CHROMA_PERSIST_DIR, DASHSCOPE_API_KEY, KNOWLEDGE_PATH
from ..schemas import FoodInfo, HealthMetrics


def extract_food_info_from_image(image_path: Path) -> FoodInfo:
    if not DASHSCOPE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务器未配置 DASHSCOPE_API_KEY 环境变量",
        )

    from dashscope import MultiModalConversation

    prompt = """
请仔细观察图片并推算食物份量。
请严格按照以下JSON格式输出，不要包含任何Markdown标记（如```json），不要输出任何其他解释性文字！
{
"food_names": ["鸡肉", "面粉"],
"weight_g": 600,
"calories": 1900
}
"""

    image_uri = f"file://{image_path.as_posix()}"
    messages = [
        {"role": "system", "content": "You are a senior nutritionist."},
        {
            "role": "user",
            "content": [{"image": image_uri}, {"text": prompt}],
        },
    ]

    try:
        response = MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model="qvq-max",
            messages=messages,
            stream=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"VLM 调用异常：{type(exc).__name__}: {exc}",
        )

    buffer = ""
    for chunk in response:
        message = chunk.output.choices[0].message
        content = message.get("content") if hasattr(message, "get") else message.content
        if not content:
            continue
        text_piece = content[0].get("text") if hasattr(content[0], "get") else None
        if text_piece:
            buffer += text_piece

    text = buffer.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"VLM 返回的 JSON 解析失败：{exc}; 原始输出：{text[:200]}",
        )
    return FoodInfo(**payload)


def build_rag_context(query: str, top_k: int = 2) -> str:
    if not DASHSCOPE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务器未配置 DASHSCOPE_API_KEY 环境变量",
        )

    from langchain_community.document_loaders import TextLoader
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings = DashScopeEmbeddings(
        dashscope_api_key=DASHSCOPE_API_KEY,
        model="text-embedding-v4",
    )

    chroma_sqlite = CHROMA_PERSIST_DIR / "chroma.sqlite3"
    if chroma_sqlite.exists():
        vectorstore = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=embeddings,
        )
    else:
        loader = TextLoader(str(KNOWLEDGE_PATH), encoding="utf-8")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
        chunks = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR),
        )

    retrieved_docs = vectorstore.similarity_search(query, k=top_k)
    return "\n".join(doc.page_content for doc in retrieved_docs)


def analyze_health_profile(
    user_profile: Dict[str, Optional[float]],
    food_info: FoodInfo,
) -> Dict[str, Any]:
    height_cm = user_profile.get("height_cm")
    weight_kg = user_profile.get("weight_kg")
    daily_steps = int(user_profile.get("daily_steps") or 0)
    step_calories = int(user_profile.get("step_calories") or 0)
    food_calories = float(food_info.calories)
    meal_activity_gap_kcal = round(food_calories - step_calories, 1)

    bmi = None
    bmi_status = "资料不足"
    if height_cm and weight_kg:
        height_m = height_cm / 100
        bmi = round(weight_kg / (height_m * height_m), 1)
        if bmi < 18.5:
            bmi_status = "偏瘦"
        elif bmi < 24:
            bmi_status = "正常"
        elif bmi < 28:
            bmi_status = "超重"
        else:
            bmi_status = "肥胖风险"

    estimated_resting_kcal = round(weight_kg * 24, 1) if weight_kg else None

    risk_tags: list[str] = []
    if food_calories >= 800:
        risk_tags.append("高热量摄入")
    if step_calories < 250:
        risk_tags.append("运动消耗偏低")
    if meal_activity_gap_kcal >= 500:
        risk_tags.append("热量缺口需补救")
    if bmi_status in {"超重", "肥胖风险"}:
        risk_tags.append("体重控制优先")
    if not risk_tags:
        risk_tags.append("总体可控")

    profile_summary_parts = []
    if height_cm:
        profile_summary_parts.append(f"身高 {height_cm} cm")
    if weight_kg:
        profile_summary_parts.append(f"体重 {weight_kg} kg")
    if bmi is not None:
        profile_summary_parts.append(f"BMI {bmi}（{bmi_status}）")
    profile_summary_parts.append(f"今日步数 {daily_steps} 步")
    profile_summary_parts.append(f"估算运动消耗 {step_calories} kcal")
    profile_summary_parts.append(f"本次饮食摄入 {food_calories} kcal")
    profile_summary = "，".join(profile_summary_parts)

    metrics = HealthMetrics(
        bmi=bmi,
        bmi_status=bmi_status,
        estimated_resting_kcal=estimated_resting_kcal,
        daily_steps=daily_steps,
        step_calories=step_calories,
        food_calories=food_calories,
        meal_activity_gap_kcal=meal_activity_gap_kcal,
    )
    return {
        "profile_summary": profile_summary,
        "metrics": metrics.model_dump(),
        "risk_tags": risk_tags,
    }


def build_personalized_rag_query(
    user_profile: Dict[str, Optional[float]],
    food_info: FoodInfo,
    analysis: Dict[str, Any],
) -> str:
    food_name = food_info.food_names[0] if food_info.food_names else "高热量饮食"
    bmi_status = analysis["metrics"]["bmi_status"]
    step_calories = int(user_profile.get("step_calories") or 0)
    activity_state = "运动不足" if step_calories < 250 else "已有一定运动基础"
    return (
        f"{food_name} 高热量饮食后如何补救；用户 BMI 状态为{bmi_status}，"
        f"当前{activity_state}，需要生成次日饮食与运动干预建议"
    )


def _extract_json_payload(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"未找到有效 JSON 对象：{text[:200]}")
    return json.loads(cleaned[start : end + 1])


def generate_health_report_llm(
    user_profile: Dict[str, Optional[float]],
    food_info: FoodInfo,
    analysis: Dict[str, Any],
    medical_context: str,
) -> Dict[str, Any]:
    if not DASHSCOPE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务器未配置 DASHSCOPE_API_KEY 环境变量",
        )

    from dashscope import MultiModalConversation

    food_name = food_info.food_names[0] if food_info.food_names else "食物"
    metrics = analysis["metrics"]
    risk_tags = "、".join(analysis["risk_tags"])
    final_prompt = f"""
【系统身份】：你是一位谨慎、专业、强调可执行性的私人营养师。

【用户概况】：
{analysis['profile_summary']}

【健康分析指标】：
{json.dumps(metrics, ensure_ascii=False)}

【风险标签】：
{risk_tags}

【饮食识别结果】：
食物：{food_name}
估算重量：{food_info.weight_g} g
估算热量：{food_info.calories} kcal

【医学参考资料】：
{medical_context}

【你的任务】：
1. 基于用户个人数据、运动数据和饮食识别结果，生成一份个性化健康报告。
2. 严格依据提供的医学参考资料组织建议，不要编造资料中没有的医学原理。
3. 输出必须是 JSON，不要输出 Markdown、解释文字或多余前后缀。

【输出 JSON 结构】：
{{
  "summary": "2-3句话的总体结论",
  "calorie_assessment": "围绕本次饮食摄入与运动消耗关系的判断",
  "diet_suggestions": ["建议1", "建议2", "建议3"],
  "exercise_suggestions": ["建议1", "建议2"],
  "cautions": ["注意事项1", "注意事项2"]
}}

【输出要求】：
- diet_suggestions 返回 2 到 4 条。
- exercise_suggestions 返回 2 到 3 条。
- cautions 返回 2 到 3 条。
- 表达自然、简洁、适合移动端直接展示。
"""

    response = MultiModalConversation.call(
        api_key=DASHSCOPE_API_KEY,
        model="qwen3.5-plus",
        messages=[
            {"role": "system", "content": "You are a senior nutritionist."},
            {"role": "user", "content": final_prompt},
        ],
        stream=False,
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"LLM 调用失败：{response.code} - {response.message}",
        )

    raw_text = response.output.choices[0].message.content[0]["text"]
    try:
        payload = _extract_json_payload(raw_text)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"健康报告 JSON 解析失败：{type(exc).__name__}: {exc}",
        )

    return {
        "summary": payload.get("summary", "").strip(),
        "calorie_assessment": payload.get("calorie_assessment", "").strip(),
        "diet_suggestions": [str(item).strip() for item in payload.get("diet_suggestions", []) if str(item).strip()],
        "exercise_suggestions": [
            str(item).strip() for item in payload.get("exercise_suggestions", []) if str(item).strip()
        ],
        "cautions": [str(item).strip() for item in payload.get("cautions", []) if str(item).strip()],
    }


def generate_health_chat_answer(
    question: str,
    user_profile: Dict[str, Optional[float]],
    medical_context: str,
    recent_report_context: Optional[str] = None,
) -> str:
    if not DASHSCOPE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务器未配置 DASHSCOPE_API_KEY 环境变量",
        )

    from dashscope import MultiModalConversation

    profile_lines = []
    if user_profile.get("height_cm") is not None:
        profile_lines.append(f"身高{user_profile['height_cm']}cm")
    if user_profile.get("weight_kg") is not None:
        profile_lines.append(f"体重{user_profile['weight_kg']}kg")
    if user_profile.get("daily_steps") is not None:
        profile_lines.append(f"今日步数{user_profile['daily_steps']}步")
    if user_profile.get("step_calories") is not None:
        profile_lines.append(f"估算消耗{user_profile['step_calories']}大卡")
    profile_text = "，".join(profile_lines) if profile_lines else "暂无可用个人数据"
    recent_report_text = recent_report_context or "暂无最近一次个性化健康报告可供参考"

    prompt = f"""
【系统身份】：你是一位谨慎、专业的健康助手。

【用户资料】：
{profile_text}

【最近一次个性化健康报告摘要】：
{recent_report_text}

【用户问题】：
{question}

【权威医学参考资料】：
{medical_context}

【回答要求】：
1. 优先依据参考资料回答，不要编造资料中没有的医学原理。
2. 如果最近一次健康报告与当前问题相关，请优先结合该报告中的结论、热量评估和建议进行解释或补充。
3. 如果用户资料不足，请明确说明回答基于一般健康建议。
4. 语言清晰、自然，适合移动端直接展示。
5. 优先给出可执行建议，不要空泛说教。
"""

    response = MultiModalConversation.call(
        api_key=DASHSCOPE_API_KEY,
        model="qwen3.5-plus",
        messages=[
            {"role": "system", "content": "You are a senior nutritionist."},
            {"role": "user", "content": prompt},
        ],
        stream=False,
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"LLM 调用失败：{response.code} - {response.message}",
        )

    return response.output.choices[0].message.content[0]["text"]


def save_upload_to_tempfile(filename: Optional[str], content: bytes) -> Path:
    suffix = Path(filename or "upload").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        temp_path = Path(handle.name)
        handle.write(content)
    return temp_path
