"""
migrate_faiss.py — Move flat FAISS indexes into per-school subdirectories.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before:  faiss_indexes/<class_name>.index
After:   faiss_indexes/<school_name>/<class_name>.index

Run ONCE after migrate_auth.py.

Usage:
  python scripts/migrate_faiss.py
"""

import os
import sys
import shutil

try:
    import mysql.connector
except ImportError:
    print("[ERROR] pip install mysql-connector-python"); sys.exit(1)

try:
    from config import db_password
except ImportError:
    db_password = "your_password"

DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": db_password,
    "database": "attendance_system",
    "autocommit": True,
}

BASE_DIR   = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
FAISS_BASE = os.path.join(BASE_DIR, "faiss_indexes")


def get_school_name(school_id: int, conn) -> str:
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT name FROM schools WHERE id = %s", (school_id,))
    row = cur.fetchone()
    cur.close()
    return row["name"] if row else "Default School"


def run() -> None:
    print("=" * 54)
    print("  Attendance System — FAISS Index Migration")
    print("=" * 54)

    conn = mysql.connector.connect(**DB_CONFIG)
    school_name = get_school_name(1, conn)
    conn.close()

    school_dir = os.path.join(FAISS_BASE, school_name)
    os.makedirs(school_dir, exist_ok=True)
    print(f"\n[FAISS] Target directory: {school_dir}")

    moved = 0
    skipped = 0

    for fname in os.listdir(FAISS_BASE):
        if not fname.endswith(".index"):
            continue
        src = os.path.join(FAISS_BASE, fname)
        dst = os.path.join(school_dir, fname)
        if os.path.exists(dst):
            print(f"[SKIP]  {fname}  (already at destination)")
            skipped += 1
            continue
        shutil.move(src, dst)
        print(f"[MOVE]  {fname}  →  {school_name}/{fname}")
        moved += 1

    print(f"\n[DONE]  Moved: {moved}   Skipped: {skipped}")
    print("\n" + "=" * 54)
    print("  FAISS migration complete.")
    print("=" * 54 + "\n")


if __name__ == "__main__":
    run()
