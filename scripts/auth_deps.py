"""
auth_deps.py — FastAPI dependency functions for JWT authentication
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage in a route:
    from auth_deps import get_current_user, require_admin, require_teacher_for_class
    
    @router.get("/protected")
    async def protected(user = Depends(get_current_user)):
        return {"user": user["full_name"]}
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

from fastapi import Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import bcrypt
from jose import JWTError, jwt

import db as DB

# ── Config ─────────────────────────────────────────────────────────────────────
SECRET_KEY   = os.environ.get("JWT_SECRET", "change-this-secret-in-production-please")
ALGORITHM    = "HS256"
TOKEN_EXPIRE_HOURS = 12

# _pwd_ctx  = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer   = HTTPBearer(auto_error=False)

# In-memory token blocklist (persists only for the process lifetime)
_revoked_tokens: set = set()


# ── Password helpers ────────────────────────────────────────────────────────────
def verify_password(plain: str, hashed: str) -> bool:
    try:
        if isinstance(hashed, str):
            hashed = hashed.encode('utf-8')
        return bcrypt.checkpw(plain.encode('utf-8'), hashed)
    except Exception:
        return False


def hash_password(plain: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain.encode('utf-8'), salt).decode('utf-8')


# ── JWT helpers ─────────────────────────────────────────────────────────────────
def create_token(data: Dict) -> str:
    payload = {**data, "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


def revoke_token(token: str) -> None:
    _revoked_tokens.add(token)


# ── DB helpers ──────────────────────────────────────────────────────────────────
def db_get_user_by_username(school_id: int, username: str) -> Optional[Dict]:
    conn = DB.get_conn()
    try:
        rows = DB.execute(
            conn,
            """SELECT u.*, s.name as school_name 
               FROM users u 
               JOIN schools s ON u.school_id = s.id
               WHERE u.school_id=%s AND u.username=%s AND u.is_active=TRUE""",
            (school_id, username),
            fetch=True,
        )
        return dict(rows[0]) if rows else None
    finally:
        conn.close()


def db_get_user_by_id(user_id: int) -> Optional[Dict]:
    conn = DB.get_conn()
    try:
        rows = DB.execute(
            conn, 
            """SELECT u.*, s.name as school_name 
               FROM users u 
               JOIN schools s ON u.school_id = s.id
               WHERE u.id=%s""", 
            (user_id,), 
            fetch=True
        )
        return dict(rows[0]) if rows else None
    finally:
        conn.close()


def db_get_teacher_class_ids(teacher_id: int):
    """Returns set of class_ids the teacher is assigned to."""
    conn = DB.get_conn()
    try:
        rows = DB.execute(
            conn,
            "SELECT class_id FROM teacher_classes WHERE teacher_id=%s",
            (teacher_id,),
            fetch=True,
        )
        return {r["class_id"] for r in rows}
    finally:
        conn.close()


# ── FastAPI Dependencies ────────────────────────────────────────────────────────
async def get_current_user(
    token: Optional[str] = Query(None),
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)
) -> Dict:
    """Dependency to get the current authenticated user from JWT (header or query param)."""
    jwt_token = None
    if creds:
        jwt_token = creds.credentials
    elif token:
        jwt_token = token
        
    if not jwt_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    if jwt_token in _revoked_tokens:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has been revoked")
    payload = decode_token(jwt_token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    user = db_get_user_by_id(int(user_id))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    user["_token"] = jwt_token  # carry token for logout
    return user


async def require_admin(user: Dict = Depends(get_current_user)) -> Dict:
    """Raises 403 if user is not an admin."""
    if user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user


def require_teacher_for_class(class_id: int, user: Dict = Depends(get_current_user)) -> Dict:
    """
    Verifies that the user is either:
    - an admin (can access any class in their school), OR
    - a teacher assigned to the given class_id.
    """
    if user["role"] == "admin":
        return user
    assigned = db_get_teacher_class_ids(user["id"])
    if class_id not in assigned:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You are not assigned to class {class_id}",
        )
    return user
