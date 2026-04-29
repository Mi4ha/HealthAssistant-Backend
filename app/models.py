from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text

from .db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    height_cm = Column(Integer, nullable=True)
    weight_kg = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class HealthReportRecord(Base):
    __tablename__ = "health_report_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    image_filename = Column(String(255), nullable=True)
    food_names = Column(Text, nullable=False)
    food_weight_g = Column(Float, nullable=False)
    food_calories = Column(Float, nullable=False)
    daily_steps = Column(Integer, nullable=False)
    step_calories = Column(Integer, nullable=False)
    used_height_cm = Column(Integer, nullable=False)
    used_weight_kg = Column(Float, nullable=False)
    medical_context = Column(Text, nullable=True)
    report = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ChatRecord(Base):
    __tablename__ = "chat_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    references = Column(Text, nullable=True)
    daily_steps = Column(Integer, nullable=True)
    step_calories = Column(Integer, nullable=True)
    used_profile = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
