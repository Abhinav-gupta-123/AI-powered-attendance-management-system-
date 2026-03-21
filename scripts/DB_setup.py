"""
Database Setup
━━━━━━━━━━━━━━
Creates all MySQL tables for the attendance management system.

Run this ONCE before anything else.
Safe to re-run — uses IF NOT EXISTS everywhere.

Tables created:
  schools    — one row per school (ready for multi-school later)
  classes    — one row per class within a school
  students   — one row per student, holds embedding BLOB
  attendance — one row per marked-present event

Usage:
  python db_setup.py

Config:
  Edit DB_CONFIG below to match your MySQL Workbench credentials.
"""

import sys

try:
    import mysql.connector
except ImportError:
    print("[ERROR] Run: pip install mysql-connector-python")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — edit these to match your MySQL Workbench setup
# ─────────────────────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",           # your MySQL username
    "password": "abhi#mysql@9981",  # your MySQL password
}

DATABASE_NAME = "attendance_system"

# ─────────────────────────────────────────────────────────────────────────────
#  SQL STATEMENTS
# ─────────────────────────────────────────────────────────────────────────────

SQL_CREATE_DB = f"""
    CREATE DATABASE IF NOT EXISTS `{DATABASE_NAME}`
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;
"""

SQL_USE_DB = f"USE `{DATABASE_NAME}`;"

SQL_SCHOOLS = """
CREATE TABLE IF NOT EXISTS schools (
    id            INT          AUTO_INCREMENT PRIMARY KEY,
    name          VARCHAR(200) NOT NULL,
    address       TEXT,
    created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

SQL_DEFAULT_SCHOOL = """
INSERT INTO schools (id, name, address)
VALUES (1, 'Default School', 'Local')
ON DUPLICATE KEY UPDATE name = name;
"""

SQL_CLASSES = """
CREATE TABLE IF NOT EXISTS classes (
    id            INT          AUTO_INCREMENT PRIMARY KEY,
    school_id     INT          NOT NULL DEFAULT 1,
    name          VARCHAR(100) NOT NULL,   -- e.g. CS-A, MECH-B, 11-Science
    batch_year    VARCHAR(10),             -- e.g. 2024
    created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP,

    UNIQUE  KEY  uq_class      (school_id, name, batch_year),
    INDEX        idx_school    (school_id),
    FOREIGN KEY  (school_id)   REFERENCES schools(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

SQL_STUDENTS = """
CREATE TABLE IF NOT EXISTS students (
    id            INT           AUTO_INCREMENT PRIMARY KEY,

    school_id     INT           NOT NULL DEFAULT 1,
    class_id      INT           NOT NULL,

    name          VARCHAR(200)  NOT NULL,
    roll_no       VARCHAR(50)   NOT NULL,

    -- Raw embedding blobs (512-dim float32 = 2048 bytes each)
    -- Stored separately per angle — same strategy as enrolled.csv
    -- All 3 added to FAISS with same student id
    -- Source of truth: FAISS index rebuilt from these blobs any time
    emb_1         MEDIUMBLOB,
    emb_2         MEDIUMBLOB,
    emb_3         MEDIUMBLOB,

    photo_path    VARCHAR(500),            -- path to enrolled_images/<id>/
    enrolled_at   DATETIME      DEFAULT CURRENT_TIMESTAMP,

    UNIQUE  KEY  uq_roll        (school_id, class_id, roll_no),
    INDEX        idx_class      (class_id),
    INDEX        idx_school     (school_id),
    FOREIGN KEY  (school_id)    REFERENCES schools(id)  ON DELETE CASCADE,
    FOREIGN KEY  (class_id)     REFERENCES classes(id)  ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

SQL_ATTENDANCE = """
CREATE TABLE IF NOT EXISTS attendance (
    id               INT         AUTO_INCREMENT PRIMARY KEY,

    student_id       INT         NOT NULL,
    class_id         INT         NOT NULL,
    school_id        INT         NOT NULL DEFAULT 1,

    date             DATE        NOT NULL,
    marked_at        DATETIME    DEFAULT CURRENT_TIMESTAMP,
    status           ENUM('present','absent','manual') DEFAULT 'present',

    -- Confidence score from FAISS cosine similarity
    -- Stored for audit — teacher can review low-confidence marks
    confidence_score FLOAT,

    session_id       VARCHAR(50),   -- groups all marks from one session

    UNIQUE  KEY  uq_daily        (student_id, date),    -- one record per student per day
    INDEX        idx_student     (student_id),
    INDEX        idx_class_date  (class_id, date),
    INDEX        idx_date        (date),
    FOREIGN KEY  (student_id)    REFERENCES students(id)  ON DELETE CASCADE,
    FOREIGN KEY  (class_id)      REFERENCES classes(id)   ON DELETE CASCADE,
    FOREIGN KEY  (school_id)     REFERENCES schools(id)   ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

SQL_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id               VARCHAR(50)  PRIMARY KEY,   -- UUID
    school_id        INT          NOT NULL DEFAULT 1,
    class_id         INT          NOT NULL,
    teacher_name     VARCHAR(200),
    started_at       DATETIME     DEFAULT CURRENT_TIMESTAMP,
    ended_at         DATETIME,
    total_present    INT          DEFAULT 0,
    total_absent     INT          DEFAULT 0,
    status           ENUM('active','completed') DEFAULT 'active',

    INDEX  idx_class    (class_id),
    INDEX  idx_date     (started_at),
    FOREIGN KEY (class_id)  REFERENCES classes(id)  ON DELETE CASCADE,
    FOREIGN KEY (school_id) REFERENCES schools(id)  ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# ─────────────────────────────────────────────────────────────────────────────
#  RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    print("=" * 54)
    print("  Attendance System — Database Setup")
    print("=" * 54)

    # ── Connect without selecting a database first ────────────────────────
    print(f"\n[DB] Connecting to MySQL at "
          f"{DB_CONFIG['host']}:{DB_CONFIG['port']} ...")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("[DB] Connected.\n")
    except mysql.connector.Error as e:
        print(f"\n[ERROR] Cannot connect to MySQL: {e}")
        print("\nCheck DB_CONFIG in this file:")
        print(f"  host     = {DB_CONFIG['host']}")
        print(f"  port     = {DB_CONFIG['port']}")
        print(f"  user     = {DB_CONFIG['user']}")
        print(f"  password = {'*' * len(DB_CONFIG['password'])}")
        sys.exit(1)

    # ── Create database ───────────────────────────────────────────────────
    print(f"[DB] Creating database '{DATABASE_NAME}' if not exists ...")
    cursor.execute(SQL_CREATE_DB)
    cursor.execute(SQL_USE_DB)
    print(f"[DB] Using database '{DATABASE_NAME}'.\n")

    # ── Create tables in dependency order ────────────────────────────────
    tables = [
        ("schools",    SQL_SCHOOLS),
        ("classes",    SQL_CLASSES),
        ("students",   SQL_STUDENTS),
        ("attendance", SQL_ATTENDANCE),
        ("sessions",   SQL_SESSIONS),
    ]

    for name, sql in tables:
        print(f"[TABLE] Creating '{name}' ...")
        cursor.execute(sql)
        print(f"[TABLE] '{name}' ready.")

    # ── Insert default school so foreign keys work immediately ───────────
    print("\n[SEED] Inserting default school (id=1) ...")
    cursor.execute(SQL_DEFAULT_SCHOOL)
    conn.commit()
    print("[SEED] Default school ready.")

    # ── Verify ────────────────────────────────────────────────────────────
    print("\n[VERIFY] Tables in database:")
    cursor.execute("SHOW TABLES;")
    for (table,) in cursor.fetchall():
        cursor.execute(f"DESCRIBE `{table}`;")
        cols = [row[0] for row in cursor.fetchall()]
        print(f"  {table:<14} → {', '.join(cols)}")

    cursor.close()
    conn.close()

    print("\n" + "=" * 54)
    print("  Setup complete. Run migrate_csv.py next.")
    print("=" * 54 + "\n")


if __name__ == "__main__":
    run()