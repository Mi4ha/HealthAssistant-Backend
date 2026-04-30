import json
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import ChatRecord, HealthReportRecord
from ..schemas import ChatHistoryItem, ChatRequest, ChatResponse
from ..services.ai import build_rag_context, generate_health_chat_answer
from ..services.auth import get_current_user_or_401
from .auth import oauth2_scheme


router = APIRouter(prefix="/chat", tags=["chat"])


def _build_recent_report_context(record: Optional[HealthReportRecord]) -> Optional[str]:
    if record is None:
        return None

    try:
        report_payload = json.loads(record.report)
    except Exception:
        report_payload = {}

    food_names = json.loads(record.food_names) if record.food_names else []
    summary = str(report_payload.get("summary", "")).strip()
    calorie_assessment = str(report_payload.get("calorie_assessment", "")).strip()
    diet_suggestions = report_payload.get("diet_suggestions", []) or []
    exercise_suggestions = report_payload.get("exercise_suggestions", []) or []
    cautions = report_payload.get("cautions", []) or []
    medical_refs = [line.strip() for line in (record.medical_context or "").splitlines() if line.strip()][:2]

    sections = [
        f"生成时间：{record.created_at}",
        f"当次食物：{'、'.join(food_names) if food_names else '未记录'}",
        f"当次摄入热量：{record.food_calories} kcal",
        f"当次运动消耗：{record.step_calories} kcal",
    ]
    if summary:
        sections.append(f"总体判断：{summary}")
    if calorie_assessment:
        sections.append(f"热量评估：{calorie_assessment}")
    if diet_suggestions:
        sections.append(f"饮食建议：{'；'.join(str(item) for item in diet_suggestions[:2])}")
    if exercise_suggestions:
        sections.append(f"运动建议：{'；'.join(str(item) for item in exercise_suggestions[:2])}")
    if cautions:
        sections.append(f"注意事项：{'；'.join(str(item) for item in cautions[:2])}")
    if medical_refs:
        sections.append(f"报告参考依据：{'；'.join(medical_refs)}")
    return "\n".join(sections)


@router.post("", response_model=ChatResponse, summary="健康问答（RAG + LLM）")
def health_chat(
    payload: ChatRequest,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    user = get_current_user_or_401(token=token, db=db)
    recent_report = (
        db.query(HealthReportRecord)
        .filter(HealthReportRecord.user_id == user.id)
        .order_by(HealthReportRecord.created_at.desc())
        .first()
    )
    recent_report_context = _build_recent_report_context(recent_report)
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
        recent_report_context=recent_report_context,
    )
    response = ChatResponse(
        answer=answer,
        references=references[:3],
        used_profile=bool(user.height_cm or user.weight_kg),
        used_recent_report=recent_report is not None,
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
            used_recent_report=False,
            created_at=record.created_at,
        )
        for record in records
    ]
