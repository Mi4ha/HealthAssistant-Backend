import json
import logging
import threading
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import HealthReportRecord
from ..schemas import HealthReportContent, HealthReportHistoryItem, HealthReportResponse
from ..services.ai import (
    analyze_health_profile,
    build_personalized_rag_query,
    build_rag_result,
    extract_food_info_from_image,
    generate_health_report_llm,
    save_upload_to_tempfile,
)
from ..services.auth import get_current_user_or_401
from .auth import oauth2_scheme


router = APIRouter(prefix="/health", tags=["health"])
logger = logging.getLogger("uvicorn.error")


def _remove_tempfile_with_retry(path):
    for attempt in range(1, 6):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(0.5 * attempt)
        except Exception:
            logger.warning("failed to remove upload tempfile: %s", path, exc_info=True)
            return
    logger.warning("upload tempfile is still locked after retries: %s", path)


def _parse_report_content(raw_report: str) -> HealthReportContent:
    try:
        payload = json.loads(raw_report)
        if isinstance(payload, dict):
            return HealthReportContent(
                summary=payload.get("summary", "").strip(),
                calorie_assessment=payload.get("calorie_assessment", "").strip(),
                diet_suggestions=payload.get("diet_suggestions", []) or [],
                exercise_suggestions=payload.get("exercise_suggestions", []) or [],
                cautions=payload.get("cautions", []) or [],
            )
    except Exception:
        pass

    return HealthReportContent(
        summary=raw_report,
        calorie_assessment="该历史记录生成于旧版本，暂不包含结构化热量评估。",
        diet_suggestions=[],
        exercise_suggestions=[],
        cautions=[],
    )


@router.post(
    "/report",
    response_model=HealthReportResponse,
    summary="上传图片并生成健康报告（VLM + RAG + LLM）",
)
def health_report(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
    image: UploadFile = File(...),
    height: Optional[int] = Form(None),
    weight: Optional[float] = Form(None),
    daily_steps: int = Form(...),
    step_calories: Optional[int] = Form(None),
):
    request_started_at = time.perf_counter()
    last_step_at = request_started_at

    def log_step(step: str) -> None:
        nonlocal last_step_at
        now = time.perf_counter()
        logger.info(
            "health_report step=%s elapsed=%.2fs total=%.2fs",
            step,
            now - last_step_at,
            now - request_started_at,
        )
        last_step_at = now

    user = get_current_user_or_401(token=token, db=db)

    height_value = height if height is not None else user.height_cm
    weight_value = weight if weight is not None else user.weight_kg

    if height_value is None or weight_value is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="缺少身高或体重信息，请先完善个人资料或在本次请求中显式传入",
        )

    step_calories_value = step_calories if step_calories is not None else 300
    user_profile = {
        "height_cm": height_value,
        "weight_kg": weight_value,
        "daily_steps": daily_steps,
        "step_calories": step_calories_value,
    }

    temp_path = None
    try:
        temp_path = save_upload_to_tempfile(image.filename, image.file.read())
        log_step("save_upload")

        food_info = extract_food_info_from_image(temp_path)
        log_step("vlm_food_detection")

        analysis = analyze_health_profile(
            user_profile=user_profile,
            food_info=food_info,
        )
        log_step("local_health_analysis")

        rag_query = build_personalized_rag_query(
            user_profile=user_profile,
            food_info=food_info,
            analysis=analysis,
        )
        rag_result = build_rag_result(rag_query, top_k=5, user_profile=user_profile)
        medical_context = rag_result.context
        log_step("rag_search")

        report_payload = generate_health_report_llm(
            user_profile=user_profile,
            food_info=food_info,
            analysis=analysis,
            medical_context=medical_context,
        )
        log_step("llm_report_generation")

        record = HealthReportRecord(
            user_id=user.id,
            image_filename=image.filename,
            food_names=json.dumps(food_info.food_names, ensure_ascii=False),
            food_weight_g=food_info.weight_g,
            food_calories=food_info.calories,
            daily_steps=daily_steps,
            step_calories=step_calories_value,
            used_height_cm=height_value,
            used_weight_kg=weight_value,
            medical_context=medical_context,
            report=json.dumps(report_payload, ensure_ascii=False),
        )
        db.add(record)
        db.commit()
        log_step("db_commit")

        return HealthReportResponse(
            food_info=food_info,
            profile_summary=analysis["profile_summary"],
            metrics=analysis["metrics"],
            risk_tags=analysis["risk_tags"],
            summary=report_payload["summary"],
            calorie_assessment=report_payload["calorie_assessment"],
            diet_suggestions=report_payload["diet_suggestions"],
            exercise_suggestions=report_payload["exercise_suggestions"],
            cautions=report_payload["cautions"],
            references=rag_result.references[:5],
        )
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except PermissionError:
                threading.Thread(
                    target=_remove_tempfile_with_retry,
                    args=(temp_path,),
                    name="upload-tempfile-cleanup",
                    daemon=True,
                ).start()
            except Exception:
                logger.warning("failed to remove upload tempfile: %s", temp_path, exc_info=True)
        logger.info("health_report finished total=%.2fs", time.perf_counter() - request_started_at)


@router.get(
    "/reports",
    response_model=list[HealthReportHistoryItem],
    summary="获取当前用户的健康报告历史",
)
def list_health_reports(
    limit: int = 20,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    user = get_current_user_or_401(token=token, db=db)
    safe_limit = max(1, min(limit, 100))
    records = (
        db.query(HealthReportRecord)
        .filter(HealthReportRecord.user_id == user.id)
        .order_by(HealthReportRecord.created_at.desc())
        .limit(safe_limit)
        .all()
    )
    return [
        HealthReportHistoryItem(
            id=record.id,
            food_names=json.loads(record.food_names),
            food_weight_g=record.food_weight_g,
            food_calories=record.food_calories,
            daily_steps=record.daily_steps,
            step_calories=record.step_calories,
            used_height_cm=record.used_height_cm,
            used_weight_kg=record.used_weight_kg,
            report=_parse_report_content(record.report),
            references=[line.strip() for line in (record.medical_context or "").splitlines() if line.strip()][:3],
            created_at=record.created_at,
        )
        for record in records
    ]
