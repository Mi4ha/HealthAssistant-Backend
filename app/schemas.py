from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    height_cm: Optional[int] = Field(default=None, ge=50, le=260)
    weight_kg: Optional[float] = Field(default=None, gt=0, le=500)


class UserOut(BaseModel):
    id: int
    username: str
    email: EmailStr
    height_cm: Optional[int] = None
    weight_kg: Optional[float] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserProfileUpdate(BaseModel):
    email: Optional[EmailStr] = None
    height_cm: Optional[int] = Field(default=None, ge=50, le=260)
    weight_kg: Optional[float] = Field(default=None, gt=0, le=500)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class FoodInfo(BaseModel):
    food_names: List[str]
    weight_g: float
    calories: float


class HealthReportResponse(BaseModel):
    food_info: FoodInfo
    report: str


class HealthReportHistoryItem(BaseModel):
    id: int
    food_names: List[str]
    food_weight_g: float
    food_calories: float
    daily_steps: int
    step_calories: int
    used_height_cm: int
    used_weight_kg: float
    report: str
    created_at: datetime


class ChatRequest(BaseModel):
    question: str
    daily_steps: Optional[int] = Field(default=None, ge=0)
    step_calories: Optional[int] = Field(default=None, ge=0)


class ChatResponse(BaseModel):
    answer: str
    references: List[str]
    used_profile: bool


class ChatHistoryItem(BaseModel):
    id: int
    question: str
    answer: str
    references: List[str]
    daily_steps: Optional[int] = None
    step_calories: Optional[int] = None
    used_profile: bool
    created_at: datetime
