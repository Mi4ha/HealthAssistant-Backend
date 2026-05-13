import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status

from ..config import CHROMA_PERSIST_DIR, DASHSCOPE_API_KEY, KNOWLEDGE_DIR
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



@dataclass
class RagSearchResult:
    context: str
    references: List[str]
    activated_scopes: List[str]


CONDITION_KEYWORDS: Dict[str, List[str]] = {
    "hypertension_nutrition": ["高血压", "血压", "降压", "收缩压", "舒张压"],
    "diabetes_nutrition": ["糖尿病", "血糖", "降糖", "胰岛素", "糖化血红蛋白"],
    "hyperlipidemia_nutrition": ["高脂血症", "血脂", "胆固醇", "甘油三酯", "降脂"],
    "gout_nutrition": ["痛风", "高尿酸", "尿酸", "嘌呤", "降尿酸"],
}

WEIGHT_MANAGEMENT_KEYWORDS = ["肥胖", "超重", "减肥", "减脂", "体重管理", "BMI", "bmi"]


def _parse_frontmatter(markdown: str) -> tuple[Dict[str, Any], str]:
    if not markdown.startswith("---"):
        return {}, markdown

    marker = "\n---"
    end = markdown.find(marker, 3)
    if end == -1:
        return {}, markdown

    raw_meta = markdown[3:end].strip().splitlines()
    body = markdown[end + len(marker) :].strip()
    metadata: Dict[str, Any] = {}
    current_list_key: Optional[str] = None
    for raw_line in raw_meta:
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("  - ") and current_list_key:
            metadata.setdefault(current_list_key, []).append(line[4:].strip().strip('"'))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"')
        current_list_key = None
        if value:
            metadata[key] = value
        else:
            metadata[key] = []
            current_list_key = key
    return metadata, body


def _normalize_metadata(metadata: Dict[str, Any], path: Path) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            normalized[key] = "; ".join(str(item) for item in value)
        else:
            normalized[key] = str(value)
    normalized.setdefault("title", path.stem)
    normalized.setdefault("source", normalized["title"])
    normalized.setdefault("source_url", "")
    normalized.setdefault("category", path.parent.name)
    normalized.setdefault("knowledge_scope", "default")
    normalized["file_path"] = str(path.relative_to(KNOWLEDGE_DIR.parent))
    return normalized


def _extract_section_title(chunk_text: str) -> str:
    for line in chunk_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    first_line = next((line.strip() for line in chunk_text.splitlines() if line.strip()), "知识片段")
    return first_line[:40]


def _knowledge_files() -> List[Path]:
    if not KNOWLEDGE_DIR.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"知识库目录不存在：{KNOWLEDGE_DIR}",
        )
    return sorted(
        path
        for path in KNOWLEDGE_DIR.glob("**/*.md")
        if path.is_file() and path.name.lower() != "readme.md"
    )


def _knowledge_signature(files: List[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(KNOWLEDGE_DIR)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _build_documents(files: List[Path]):
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n## ", "\n# ", "\n\n", "\n", "。", "；", "，"],
    )
    documents = []
    for path in files:
        raw = path.read_text(encoding="utf-8")
        metadata, body = _parse_frontmatter(raw)
        normalized_meta = _normalize_metadata(metadata, path)
        chunks = splitter.split_text(body)
        for index, chunk in enumerate(chunks):
            section = _extract_section_title(chunk)
            documents.append(
                Document(
                    page_content=chunk.strip(),
                    metadata={
                        **normalized_meta,
                        "section": section,
                        "chunk_index": index,
                    },
                )
            )
    if not documents:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="知识库目录中没有可用 Markdown 文档",
        )
    return documents


def _ensure_vectorstore(embeddings):
    from langchain_community.vectorstores import Chroma

    files = _knowledge_files()
    signature = _knowledge_signature(files)
    manifest_path = CHROMA_PERSIST_DIR / "knowledge_manifest.json"
    current_manifest = None
    if manifest_path.exists():
        try:
            current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            current_manifest = None

    needs_rebuild = (
        not (CHROMA_PERSIST_DIR / "chroma.sqlite3").exists()
        or not current_manifest
        or current_manifest.get("signature") != signature
        or current_manifest.get("schema_version") != 2
    )
    if needs_rebuild:
        if CHROMA_PERSIST_DIR.exists():
            shutil.rmtree(CHROMA_PERSIST_DIR)
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        vectorstore = Chroma.from_documents(
            documents=_build_documents(files),
            embedding=embeddings,
            persist_directory=str(CHROMA_PERSIST_DIR),
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "signature": signature,
                    "document_count": len(files),
                    "files": [str(path.relative_to(KNOWLEDGE_DIR)) for path in files],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return vectorstore

    return Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
    )


def _activated_categories(query: str, user_profile: Optional[Dict[str, Optional[float]]] = None) -> List[str]:
    categories = [
        "dietary_general",
        "physical_activity",
        "health_literacy",
        "nutrition_reference_intake",
        "safety",
    ]
    normalized_query = query.lower()
    bmi = None
    if user_profile:
        height_cm = user_profile.get("height_cm")
        weight_kg = user_profile.get("weight_kg")
        if height_cm and weight_kg:
            bmi = float(weight_kg) / ((float(height_cm) / 100) ** 2)
    if bmi is not None and bmi >= 24:
        categories.append("obesity_nutrition")
    elif any(keyword.lower() in normalized_query for keyword in WEIGHT_MANAGEMENT_KEYWORDS):
        categories.append("obesity_nutrition")

    for category, keywords in CONDITION_KEYWORDS.items():
        if any(keyword.lower() in normalized_query for keyword in keywords):
            categories.append(category)
    return list(dict.fromkeys(categories))


def _reference_from_metadata(metadata: Dict[str, Any]) -> str:
    title = str(metadata.get("title") or metadata.get("source") or "知识来源")
    section = str(metadata.get("section") or "相关章节")
    source_url = str(metadata.get("source_url") or "").strip()
    if source_url:
        return f"{title} - {section}（{source_url}）"
    return f"{title} - {section}"


def _format_context_item(index: int, page_content: str, metadata: Dict[str, Any]) -> str:
    title = str(metadata.get("title") or "知识来源")
    section = str(metadata.get("section") or "相关章节")
    scope = str(metadata.get("knowledge_scope") or "default")
    return (
        f"[R{index}] 来源：{title}\n"
        f"章节：{section}\n"
        f"知识范围：{scope}\n"
        f"内容：{page_content.strip()}"
    )


def build_rag_result(
    query: str,
    top_k: int = 5,
    user_profile: Optional[Dict[str, Optional[float]]] = None,
) -> RagSearchResult:
    if not DASHSCOPE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务器未配置 DASHSCOPE_API_KEY 环境变量",
        )

    from langchain_community.embeddings import DashScopeEmbeddings

    embeddings = DashScopeEmbeddings(
        dashscope_api_key=DASHSCOPE_API_KEY,
        model="text-embedding-v4",
    )
    vectorstore = _ensure_vectorstore(embeddings)
    categories = _activated_categories(query, user_profile)
    search_filter = {"category": {"$in": categories}}
    requested = max(top_k * 3, top_k)

    try:
        docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
            query,
            k=requested,
            filter=search_filter,
        )
    except Exception:
        docs_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=requested * 2)
        docs_with_scores = [
            (doc, score)
            for doc, score in docs_with_scores
            if str(doc.metadata.get("category") or "") in categories
        ]

    selected = []
    seen_refs = set()
    for doc, score in docs_with_scores:
        ref = _reference_from_metadata(doc.metadata)
        if ref in seen_refs:
            continue
        seen_refs.add(ref)
        selected.append((doc, score))
        if len(selected) >= top_k:
            break

    if not selected:
        return RagSearchResult(
            context="当前知识库没有检索到足够相关的权威片段。请基于一般健康原则保守回答，并提示必要时咨询专业人员。",
            references=[],
            activated_scopes=categories,
        )

    context_parts = [
        _format_context_item(index, doc.page_content, doc.metadata)
        for index, (doc, _) in enumerate(selected, start=1)
    ]
    references = [_reference_from_metadata(doc.metadata) for doc, _ in selected]
    return RagSearchResult(
        context="\n\n".join(context_parts),
        references=references,
        activated_scopes=categories,
    )


def build_rag_context(
    query: str,
    top_k: int = 5,
    user_profile: Optional[Dict[str, Optional[float]]] = None,
) -> str:
    return build_rag_result(query=query, top_k=top_k, user_profile=user_profile).context


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
1. 基于用户个人数据、运动数据、饮食识别结果和 [R] 编号权威参考片段，生成一份个性化健康报告。
2. 优先依据参考资料组织建议，不要编造资料中没有的医学原理。
3. 本系统是健康管理助手，不做疾病诊断，不提供药物调整或治疗方案。
4. 如果参考资料不足以支持某个判断，请用保守生活方式建议表达。
5. 输出必须是 JSON，不要输出 Markdown、解释文字或多余前后缀。

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
- cautions 中至少包含 1 条安全边界或就医提醒，避免过度医疗化。
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
1. 优先依据 [R] 编号参考资料回答，不要编造资料中没有的医学原理。
2. 如果最近一次健康报告与当前问题相关，请结合该报告中的结论、热量评估和建议进行解释或补充。
3. 如果用户资料不足，请明确说明回答基于一般健康建议。
4. 如果问题涉及疾病、检查指标或用药，只做健康教育和就医提醒，不诊断、不调药。
5. 如果参考资料不足，请说明“当前资料不足”，并给出保守建议或建议咨询专业人员。
6. 语言清晰、自然，适合移动端直接展示；优先给出可执行建议，不要空泛说教。
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
