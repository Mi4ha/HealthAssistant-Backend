import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./health_assistant.db")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-281cb6a56a954bfea647a2d1e6e0ee49").strip()
CHROMA_PERSIST_DIR = BASE_DIR / "local_chroma_db"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

