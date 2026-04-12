from datetime import datetime, timedelta
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext


DATABASE_URL = "sqlite:///./health_assistant.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    email: EmailStr
    created_at: datetime

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class FoodInfo(BaseModel):
    food_names: list[str]
    weight_g: float
    calories: float


class HealthReportResponse(BaseModel):
    food_info: FoodInfo
    report: str


app = FastAPI(title="Health Assistant Backend")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(username: str) -> str:
    # 简化处理：真实项目中应使用 JWT 并设置过期时间
    return f"token-{username}"


@app.post("/register", response_model=UserOut, summary="用户注册")
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    existing_user = (
        db.query(User)
        .filter((User.username == user_in.username) | (User.email == user_in.email))
        .first()
    )
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="用户名或邮箱已被注册"
        )

    user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/login", response_model=Token, summary="用户登录")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(username=user.username)
    return Token(access_token=access_token)


@app.get("/me", response_model=UserOut, summary="获取当前用户信息")
def read_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
):
    if not token.startswith("token-"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的访问令牌"
        )
    username = token.replace("token-", "", 1)
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在"
        )
    return user


def get_current_user_or_401(token: str, db: Session) -> User:
    if not token or not token.startswith("token-"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的访问令牌"
        )
    username = token.replace("token-", "", 1)
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在"
        )
    return user


def extract_food_info_from_image(image_path: Path) -> FoodInfo:
    """
    使用多模态模型识别食物，并要求模型输出严格 JSON。
    """
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

    # dashscope 要求 image 以 file:// 形式引用
    image_uri = f"file://{image_path.as_posix()}"
    messages = [
        {"role": "system", "content": "You are a senior nutritionist."},
        {
            "role": "user",
            "content": [
                {"image": image_uri},
                {"text": prompt},
            ],
        },
    ]

    try:
        response = MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model="qvq-max",
            messages=messages,
            stream=True,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"VLM 调用异常：{type(e).__name__}: {e}",
        )

    # stream=True 时 response 是 generator，需要手动拼接最终文本
    buf = ""
    for chunk in response:
        message = chunk.output.choices[0].message
        content = message.get("content") if hasattr(message, "get") else message.content
        if not content:
            continue
        # content 通常形如：[{ "text": "..." }]
        text_piece = content[0].get("text") if hasattr(content[0], "get") else None
        if text_piece:
            buf += text_piece

    text = buf.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail=f"VLM 返回的 JSON 解析失败：{e}; 原始输出：{text[:200]}",
        )
    return FoodInfo(**obj)


def build_rag_context(query: str, top_k: int = 2) -> str:
    """
    使用 Chroma 做相似度检索，从本地知识库拼接上下文。
    注意：这里尽量复用已持久化的向量库，避免每次请求都重新建库。
    """
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_community.vectorstores import Chroma

    if not DASHSCOPE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务器未配置 DASHSCOPE_API_KEY 环境变量",
        )

    base_dir = Path(__file__).resolve().parent
    persist_dir = base_dir / "local_chroma_db"
    knowledge_path = base_dir / "knowledge.txt"

    embeddings = DashScopeEmbeddings(
        dashscope_api_key=DASHSCOPE_API_KEY,
        model="text-embedding-v4",
    )

    chroma_sqlite = persist_dir / "chroma.sqlite3"
    if chroma_sqlite.exists():
        vectorstore = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )
    else:
        # 首次构建（或本地向量库被清空时）
        loader = TextLoader(str(knowledge_path), encoding="utf-8")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150, chunk_overlap=20
        )
        chunks = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir),
        )

    retrieved_docs = vectorstore.similarity_search(query, k=top_k)
    rag_context = "\n".join(doc.page_content for doc in retrieved_docs)
    return rag_context


def generate_health_report_llm(
    user_profile: dict,
    food_info: FoodInfo,
    medical_context: str,
) -> str:
    from dashscope import MultiModalConversation

    food_name = food_info.food_names[0] if food_info.food_names else "食物"
    food_calories = food_info.calories

    final_prompt = f"""
【系统身份】：你是一位严厉且专业的私人营养师。

【用户身体数据】：身高{user_profile['height']}cm，体重{user_profile['weight']}kg。
【今日运动数据】：今日已步行{user_profile['daily_steps']}步，约消耗{user_profile['step_calories']}大卡。
【今日超标饮食】：用户刚才吃了一顿{food_name}，摄入了高达{food_calories}大卡的热量。

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


@app.post(
    "/health/report",
    response_model=HealthReportResponse,
    summary="上传图片并生成健康报告（VLM + RAG + LLM）",
)
def health_report(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
    image: UploadFile = File(...),
    height: int = Form(...),
    weight: int = Form(...),
    daily_steps: int = Form(...),
    step_calories: Optional[int] = Form(None),
):
    # 校验登录状态
    _user = get_current_user_or_401(token=token, db=db)

    step_calories_value = step_calories if step_calories is not None else 300
    user_profile = {
        "height": height,
        "weight": weight,
        "daily_steps": daily_steps,
        "step_calories": step_calories_value,
    }

    # 保存临时文件，给 dashscope 使用 file://
    suffix = Path(image.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        temp_path = Path(f.name)
        content = image.file.read()
        f.write(content)

    try:
        food_info = extract_food_info_from_image(temp_path)
        # 用识别到的食物名作为检索词
        food_name_for_query = food_info.food_names[0] if food_info.food_names else "食物"
        rag_query = f"{food_name_for_query}吃多了，高脂高热量饮食后如何补救？"
        medical_context = build_rag_context(rag_query, top_k=2)
        report = generate_health_report_llm(
            user_profile=user_profile,
            food_info=food_info,
            medical_context=medical_context,
        )
        return HealthReportResponse(food_info=food_info, report=report)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

