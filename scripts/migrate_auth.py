"""
migrate_auth.py — Phase 2 DB migration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Adds:
  • users table           (admins + teachers per school)
  • teacher_classes table (which teachers teach which classes)
  • teacher_id column     (on sessions table)
  • Default admin account (username: admin, password: admin123)

Safe to run multiple times — uses IF NOT EXISTS / IGNORE everywhere.

Usage:
  python scripts/migrate_auth.py
"""

import sys
import hashlib

try:
    import mysql.connector
except ImportError:
    print("[ERROR] pip install mysql-connector-python"); sys.exit(1)

try:
    from passlib.context import CryptContext
    _pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
    def hash_pw(pw): return _pwd_ctx.hash(pw)
except ImportError:
    # fallback if passlib isn't installed yet
    def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

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

SQL_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id            INT           AUTO_INCREMENT PRIMARY KEY,
    school_id     INT           NOT NULL DEFAULT 1,
    username      VARCHAR(100)  NOT NULL,
    password_hash VARCHAR(256)  NOT NULL,
    full_name     VARCHAR(200)  NOT NULL,
    role          ENUM('admin','teacher') DEFAULT 'teacher',
    is_active     BOOLEAN       DEFAULT TRUE,
    created_at    DATETIME      DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_user (school_id, username),
    FOREIGN KEY (school_id) REFERENCES schools(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

SQL_TEACHER_CLASSES = """
CREATE TABLE IF NOT EXISTS teacher_classes (
    teacher_id  INT NOT NULL,
    class_id    INT NOT NULL,
    PRIMARY KEY (teacher_id, class_id),
    FOREIGN KEY (teacher_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (class_id)   REFERENCES classes(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

SQL_ADD_TEACHER_ID = """
ALTER TABLE sessions
ADD COLUMN IF NOT EXISTS teacher_id INT DEFAULT NULL,
ADD CONSTRAINT fk_sessions_teacher
    FOREIGN KEY (teacher_id) REFERENCES users(id) ON DELETE SET NULL;
"""

# Fallback if DB doesn't support IF NOT EXISTS in ALTER
SQL_CHECK_TEACHER_ID = """
SELECT COUNT(*) AS cnt
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = 'attendance_system'
  AND TABLE_NAME   = 'sessions'
  AND COLUMN_NAME  = 'teacher_id';
"""

SQL_ADD_TEACHER_ID_SIMPLE = """
ALTER TABLE sessions
ADD COLUMN teacher_id INT DEFAULT NULL,
ADD CONSTRAINT fk_sessions_teacher
    FOREIGN KEY (teacher_id) REFERENCES users(id) ON DELETE SET NULL;
"""


def run() -> None:
    print("=" * 54)
    print("  Attendance System — Auth Migration (Phase 2)")
    print("=" * 54)

    conn = mysql.connector.connect(**DB_CONFIG)
    cur  = conn.cursor(dictionary=True)

    # 1. users
    print("\n[TABLE] Creating 'users' ...")
    cur.execute(SQL_USERS)
    print("[TABLE] 'users' ready.")

    # 2. teacher_classes
    print("[TABLE] Creating 'teacher_classes' ...")
    cur.execute(SQL_TEACHER_CLASSES)
    print("[TABLE] 'teacher_classes' ready.")

    # 3. teacher_id column on sessions (safe check)
    print("[MIGRATE] Adding teacher_id to sessions ...")
    cur.execute(SQL_CHECK_TEACHER_ID)
    row = cur.fetchone()
    if row and row["cnt"] == 0:
        try:
            cur.execute(SQL_ADD_TEACHER_ID_SIMPLE)
            print("[MIGRATE] teacher_id column added.")
        except mysql.connector.Error as e:
            print(f"[MIGRATE] Skipped (already exists or unsupported): {e}")
    else:
        print("[MIGRATE] teacher_id column already exists — skipped.")

    # 4. Default admin account
    admin_hash = hash_pw("admin123")
    print("\n[SEED] Creating default admin account ...")
    try:
        cur.execute(
            """INSERT IGNORE INTO users (school_id, username, password_hash, full_name, role)
               VALUES (1, 'admin', %s, 'System Administrator', 'admin')""",
            (admin_hash,)
        )
        print("[SEED] Admin account ready (username=admin, password=admin123).")
        print("[SEED] ⚠  Change the default password immediately in production!")
    except mysql.connector.Error as e:
        print(f"[SEED] Admin already exists or error: {e}")

    cur.close()
    conn.close()

    print("\n" + "=" * 54)
    print("  Migration complete.")
    print("=" * 54 + "\n")


if __name__ == "__main__":
    run()
