import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import ChatRecord
from ..schemas import ChatHistoryItem, ChatRequest, ChatResponse
from ..services.ai import build_rag_context, generate_health_chat_answer
from ..services.auth import get_current_user_or_401
from .auth import oauth2_scheme


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, summary="健康问答（RAG + LLM）")
def health_chat(
    payload: ChatRequest,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    user = get_current_user_or_401(token=token, db=db)
    medical_context = build_rag_context(payload.question, top_k=3)
    references = [line.strip() for line in medical_context.splitlines() if line.strip()]

    user_profile = {
        "height_cm": user.height_cm,
        "weight_kg": user.weight_kg,
        "daily_steps": payload.daily_steps,
        "step_calories": payload.step_calories,
    }
    answer = generate_health_chat_answer(
        question=payload.question,
        user_profile=user_profile,
        medical_context=medical_context,
    )
    response = ChatResponse(
        answer=answer,
        references=references[:3],
        used_profile=bool(user.height_cm or user.weight_kg),
    )
    record = ChatRecord(
        user_id=user.id,
        question=payload.question,
        answer=response.answer,
        references=json.dumps(response.references, ensure_ascii=False),
        daily_steps=payload.daily_steps,
        step_calories=payload.step_calories,
        used_profile=1 if response.used_profile else 0,
    )
    db.add(record)
    db.commit()
    return response


@router.get("/history", response_model=list[ChatHistoryItem], summary="获取当前用户问答历史")
def list_chat_history(
    limit: int = 20,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    user = get_current_user_or_401(token=token, db=db)
    safe_limit = max(1, min(limit, 100))
    records = (
        db.query(ChatRecord)
        .filter(ChatRecord.user_id == user.id)
        .order_by(ChatRecord.created_at.desc())
        .limit(safe_limit)
        .all()
    )
    return [
        ChatHistoryItem(
            id=record.id,
            question=record.question,
            answer=record.answer,
            references=json.loads(record.references) if record.references else [],
            daily_steps=record.daily_steps,
            step_calories=record.step_calories,
            used_profile=bool(record.used_profile),
            created_at=record.created_at,
        )
        for record in records
    ]
