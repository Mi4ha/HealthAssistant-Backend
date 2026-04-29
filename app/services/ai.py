import json
import tempfile
from pathlib import Path
from typing import Dict, Optional

from fastapi import HTTPException, status

from ..config import CHROMA_PERSIST_DIR, DASHSCOPE_API_KEY, KNOWLEDGE_PATH
from ..schemas import FoodInfo


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


def generate_health_report_llm(
    user_profile: Dict[str, Optional[float]],
    food_info: FoodInfo,
    medical_context: str,
) -> str:
    if not DASHSCOPE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务器未配置 DASHSCOPE_API_KEY 环境变量",
        )

    from dashscope import MultiModalConversation

    food_name = food_info.food_names[0] if food_info.food_names else "食物"
    final_prompt = f"""
【系统身份】：你是一位严厉且专业的私人营养师。

【用户身体数据】：身高{user_profile['height_cm']}cm，体重{user_profile['weight_kg']}kg。
【今日运动数据】：今日已步行{user_profile['daily_steps']}步，约消耗{user_profile['step_calories']}大卡。
【今日超标饮食】：用户刚才吃了一顿{food_name}，摄入了高达{food_info.calories}大卡的热量。

【权威医学参考资料】：
{medical_context}

【你的任务】：
1. 评估用户今天的热量盈亏情况（结合他的运动消耗）。
2. 严格依据上述提供的[权威医学参考资料]，为用户制定明天的饮食和运动补救计划。不要编造资料中没有的医学原理。
3. 语气要像专业的私人教练，适当严厉，分点给出建议。
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

    return response.output.choices[0].message.content[0]["text"]


def generate_health_chat_answer(
    question: str,
    user_profile: Dict[str, Optional[float]],
    medical_context: str,
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

    prompt = f"""
【系统身份】：你是一位谨慎、专业的健康助手。

【用户资料】：
{profile_text}

【用户问题】：
{question}

【权威医学参考资料】：
{medical_context}

【回答要求】：
1. 优先依据参考资料回答，不要编造资料中没有的医学原理。
2. 如果用户资料不足，请明确说明回答基于一般健康建议。
3. 语言清晰、自然，适合移动端直接展示。
4. 优先给出可执行建议，不要空泛说教。
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
