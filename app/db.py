from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import DATABASE_URL


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _migrate_users_table()


def _migrate_users_table():
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("users")}
    statements = []
    if "height_cm" not in existing_columns:
        statements.append("ALTER TABLE users ADD COLUMN height_cm INTEGER")
    if "weight_kg" not in existing_columns:
        statements.append("ALTER TABLE users ADD COLUMN weight_kg REAL")
    if "updated_at" not in existing_columns:
        statements.append("ALTER TABLE users ADD COLUMN updated_at DATETIME")

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
        if "updated_at" not in existing_columns:
            connection.execute(
                text(
                    "UPDATE users SET updated_at = created_at WHERE updated_at IS NULL"
                )
            )

