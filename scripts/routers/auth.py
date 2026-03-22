"""
routers/auth.py — Authentication router
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Routes:
  POST /auth/login   — returns JWT
  POST /auth/logout  — revokes current token
  GET  /auth/me      — returns current user info
  POST /auth/users   — create new user (admin only)
  GET  /auth/users   — list all users in school (admin only)
  PUT  /auth/users/{uid}/classes — assign teacher to classes (admin only)
  GET  /auth/classes — list all classes in school (for assignment UI)
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import os

import db as DB
from auth_deps import (
    verify_password, hash_password, create_token, revoke_token,
    db_get_user_by_username, db_get_user_by_id,
    get_current_user, require_admin,
)

router = APIRouter(prefix="/auth", tags=["auth"])

# ── Serve login.html ───────────────────────────────────────────────────────────
_FRONTEND = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))

def _serve(fname: str) -> HTMLResponse:
    p = os.path.join(_FRONTEND, fname)
    try:
        return HTMLResponse(open(p, encoding="utf-8").read())
    except FileNotFoundError:
        return HTMLResponse(f"<h2>Missing: {fname}</h2>", status_code=404)

@router.get("/ui/login", tags=["ui"], response_class=HTMLResponse)
async def ui_login():
    return _serve("login.html")


# ── Request / Response Models ──────────────────────────────────────────────────
class LoginRequest(BaseModel):
    school_id: int = 1
    username:  str
    password:  str


class UserCreate(BaseModel):
    school_id:  int = 1
    username:   str
    password:   str
    full_name:  str
    role:       str = "teacher"   # 'admin' or 'teacher'


class AssignClassesRequest(BaseModel):
    class_ids: List[int]


# ── Routes ─────────────────────────────────────────────────────────────────────
@router.post("/login")
async def login(req: LoginRequest):
    user = db_get_user_by_username(req.school_id, req.username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_token({"sub": str(user["id"]), "role": user["role"], "school_id": user["school_id"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id":        user["id"],
            "username":  user["username"],
            "full_name": user["full_name"],
            "role":      user["role"],
            "school_id": user["school_id"],
        },
    }


@router.post("/logout")
async def logout(user=Depends(get_current_user)):
    revoke_token(user["_token"])
    return {"detail": "Logged out successfully"}


@router.get("/me")
async def me(user=Depends(get_current_user)):
    return {
        "id":        user["id"],
        "username":  user["username"],
        "full_name": user["full_name"],
        "role":      user["role"],
        "school_id": user["school_id"],
    }


# ── Admin: User Management ─────────────────────────────────────────────────────
@router.post("/users", status_code=201)
async def create_user(req: UserCreate, admin=Depends(require_admin)):
    if req.role not in ("admin", "teacher"):
        raise HTTPException(status_code=400, detail="role must be 'admin' or 'teacher'")
    pw_hash = hash_password(req.password)
    conn = DB.get_conn()
    try:
        DB.execute(
            conn,
            "INSERT INTO users (school_id, username, password_hash, full_name, role) VALUES (%s,%s,%s,%s,%s)",
            (req.school_id, req.username, pw_hash, req.full_name, req.role),
        )
        rows = DB.execute(conn, "SELECT id FROM users WHERE school_id=%s AND username=%s",
                          (req.school_id, req.username), fetch=True)
        return {"id": rows[0]["id"], "username": req.username, "role": req.role}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()


@router.get("/users")
async def list_users(admin=Depends(require_admin)):
    conn = DB.get_conn()
    try:
        rows = DB.execute(
            conn,
            "SELECT id, username, full_name, role, is_active, created_at FROM users WHERE school_id=%s ORDER BY role, full_name",
            (admin["school_id"],),
            fetch=True,
        )
        return rows
    finally:
        conn.close()


@router.put("/users/{uid}/classes")
async def assign_classes(uid: int, req: AssignClassesRequest, admin=Depends(require_admin)):
    """Replace teacher's class assignments with the provided list."""
    conn = DB.get_conn()
    try:
        # Verify user belongs to admin's school
        user = db_get_user_by_id(uid)
        if not user or user["school_id"] != admin["school_id"]:
            raise HTTPException(status_code=404, detail="User not found")
        if user["role"] != "teacher":
            raise HTTPException(status_code=400, detail="Can only assign classes to teachers")
        # Clear old assignments + insert new ones
        DB.execute(conn, "DELETE FROM teacher_classes WHERE teacher_id=%s", (uid,))
        for cid in req.class_ids:
            try:
                DB.execute(conn, "INSERT IGNORE INTO teacher_classes (teacher_id, class_id) VALUES (%s,%s)",
                           (uid, cid))
            except Exception:
                pass
        return {"teacher_id": uid, "class_ids": req.class_ids}
    finally:
        conn.close()


@router.get("/classes")
async def list_classes(admin=Depends(require_admin)):
    """List all classes in the school (for admin assignment UI)."""
    conn = DB.get_conn()
    try:
        rows = DB.execute(
            conn,
            "SELECT id, name, batch_year FROM classes WHERE school_id=%s ORDER BY name",
            (admin["school_id"],),
            fetch=True,
        )
        return rows
    finally:
        conn.close()
