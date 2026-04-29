from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import User
from ..schemas import Token, UserCreate, UserOut, UserProfileUpdate
from ..services.auth import (
    authenticate_user,
    create_access_token,
    get_current_user_or_401,
    get_password_hash,
)


router = APIRouter(tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


@router.post("/register", response_model=UserOut, summary="用户注册")
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    existing_user = (
        db.query(User)
        .filter((User.username == user_in.username) | (User.email == user_in.email))
        .first()
    )
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名或邮箱已被注册",
        )

    user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),
        height_cm=user_in.height_cm,
        weight_kg=user_in.weight_kg,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=Token, summary="用户登录")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
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


@router.get("/me", response_model=UserOut, summary="获取当前用户信息")
def read_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    return get_current_user_or_401(token=token, db=db)


@router.patch("/me/profile", response_model=UserOut, summary="更新当前用户资料")
def update_current_user_profile(
    payload: UserProfileUpdate,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    user = get_current_user_or_401(token=token, db=db)

    if payload.email is not None and payload.email != user.email:
        existing_email = db.query(User).filter(User.email == payload.email).first()
        if existing_email and existing_email.id != user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="邮箱已被其他账号使用",
            )
        user.email = payload.email

    if payload.height_cm is not None:
        user.height_cm = payload.height_cm
    if payload.weight_kg is not None:
        user.weight_kg = payload.weight_kg

    db.add(user)
    db.commit()
    db.refresh(user)
    return user

