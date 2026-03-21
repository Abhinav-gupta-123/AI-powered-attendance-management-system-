"""
Migration Script — enrolled.csv → MySQL + FAISS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reads existing enrolled.csv, writes every student into MySQL,
then builds per-class FAISS IndexIDMap files on disk.

Run ONCE after db_setup.py.
Safe to re-run — skips students already in MySQL by roll_no.

What it does:
  1. Reads enrolled.csv (handles all previous formats)
  2. For each student:
       a. Asks which class they belong to (or creates the class)
       b. MySQL INSERT → gets permanent student id
       c. Adds all 3 embeddings to that class FAISS index
  3. Saves all FAISS .index files to faiss_indexes/<class_name>.index

Usage:
  python migrate_csv.py

After this script:
  - MySQL has every student with embedding BLOBs
  - faiss_indexes/ has one .index file per class
  - enrolled.csv is kept as backup — not deleted
"""

import csv
import os
import sys
import struct
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import mysql.connector
except ImportError:
    print("[ERROR] Run: pip install mysql-connector-python")
    sys.exit(1)

try:
    import faiss
except ImportError:
    print("[ERROR] Run: pip install faiss-cpu")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — must match db_setup.py
# ─────────────────────────────────────────────────────────────────────────────

DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": "abhi#mysql@9981",
    "database": "attendance_system",
}

SCHOOL_ID      = 1               # default school
ENROLLED_CSV   = "enrolled.csv"  # your existing file
FAISS_DIR      = "faiss_indexes" # folder for .index files
EMBEDDING_DIM  = 512             # buffalo_l / AdaFace both output 512-dim


# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_emb(raw: str) -> Optional[np.ndarray]:
    """Parse pipe-separated string → 512-dim float32 array."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        arr = np.array([float(x) for x in raw.split("|")],
                       dtype=np.float32)
        return arr if len(arr) == EMBEDDING_DIM else None
    except Exception:
        return None


def emb_to_blob(emb: np.ndarray) -> bytes:
    """512-dim float32 → raw bytes for MySQL MEDIUMBLOB."""
    return emb.astype(np.float32).tobytes()


def blob_to_emb(blob: bytes) -> np.ndarray:
    """Raw bytes from MySQL MEDIUMBLOB → 512-dim float32."""
    return np.frombuffer(blob, dtype=np.float32).copy()


# ─────────────────────────────────────────────────────────────────────────────
#  CSV READER — handles all previous formats
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> List[Dict]:
    """
    Returns list of:
      { csv_student_id, name, embeddings: [np.ndarray, ...] }
    Handles emb_1/emb_2/emb_3, embeddings (||), and single embedding formats.
    """
    records = []
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found.")
        sys.exit(1)

    with open(path, "r", newline="") as f:
        reader  = csv.DictReader(f)
        headers = list(reader.fieldnames or [])

        for row in reader:
            embs = []
            if "emb_1" in headers:
                for key in ["emb_1", "emb_2", "emb_3"]:
                    e = parse_emb(str(row.get(key, "")))
                    if e is not None:
                        embs.append(e)
            elif "embeddings" in headers:
                for block in str(row.get("embeddings", "")).split("||"):
                    e = parse_emb(block)
                    if e is not None:
                        embs.append(e)
            elif "embedding" in headers:
                e = parse_emb(str(row.get("embedding", "")))
                if e is not None:
                    embs.append(e)

            if embs:
                records.append({
                    "csv_student_id": str(row.get("student_id", "")),
                    "name":           str(row.get("name", "")),
                    "embeddings":     embs,
                })
            else:
                print(f"[WARN] No valid embeddings for "
                      f"{row.get('student_id')} — skipping")

    print(f"[CSV] Loaded {len(records)} student(s) from {path}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
#  MYSQL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_class(cursor, school_id: int,
                         class_name: str) -> int:
    """
    Returns class id for given class_name.
    Creates the class row if it doesn't exist.
    """
    cursor.execute(
        "SELECT id FROM classes WHERE school_id=%s AND name=%s LIMIT 1",
        (school_id, class_name)
    )
    row = cursor.fetchone()
    if row:
        return int(row[0])

    cursor.execute(
        "INSERT INTO classes (school_id, name) VALUES (%s, %s)",
        (school_id, class_name)
    )
    return cursor.lastrowid


def student_exists(cursor, school_id: int,
                   class_id: int, roll_no: str) -> Optional[int]:
    """Returns student id if already in MySQL, else None."""
    cursor.execute(
        """SELECT id FROM students
           WHERE school_id=%s AND class_id=%s AND roll_no=%s
           LIMIT 1""",
        (school_id, class_id, roll_no)
    )
    row = cursor.fetchone()
    return int(row[0]) if row else None


def insert_student(cursor, school_id: int, class_id: int,
                   name: str, roll_no: str,
                   embeddings: List[np.ndarray],
                   photo_path: str) -> int:
    """
    Inserts student into MySQL.
    Returns the AUTO_INCREMENT id — this is the permanent FAISS id.
    """
    blobs = [emb_to_blob(e) for e in embeddings]
    while len(blobs) < 3:
        blobs.append(None)

    cursor.execute(
        """INSERT INTO students
           (school_id, class_id, name, roll_no,
            emb_1, emb_2, emb_3, photo_path)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        (school_id, class_id, name, roll_no,
         blobs[0], blobs[1], blobs[2], photo_path)
    )
    return cursor.lastrowid


# ─────────────────────────────────────────────────────────────────────────────
#  FAISS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_or_create_index(path: str) -> faiss.IndexIDMap:
    """
    Load existing FAISS index from disk, or create a new one.
    Uses IndexIDMap wrapping IndexFlatIP (inner product = cosine on L2-normed).
    """
    if os.path.exists(path):
        index = faiss.read_index(path)
        print(f"[FAISS] Loaded existing index: {path} "
              f"({index.ntotal} vectors)")
        return index

    base  = faiss.IndexFlatIP(EMBEDDING_DIM)
    index = faiss.IndexIDMap(base)
    print(f"[FAISS] Created new index: {path}")
    return index


def add_to_index(index: faiss.IndexIDMap,
                 embeddings: List[np.ndarray],
                 student_id: int) -> None:
    """
    Add all embeddings for one student to the FAISS index.
    All 3 angles stored with the same student_id.
    FAISS returns this id on search — maps directly to MySQL students.id
    """
    for emb in embeddings:
        # L2-normalise before adding (required for cosine similarity
        # via IndexFlatIP — dot product of unit vectors = cosine sim)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        vec = emb.reshape(1, -1).astype(np.float32)
        ids = np.array([student_id], dtype=np.int64)
        index.add_with_ids(vec, ids)


def save_index(index: faiss.IndexIDMap, path: str) -> None:
    faiss.write_index(index, path)
    print(f"[FAISS] Saved: {path}  ({index.ntotal} vectors)")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN MIGRATION
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    print("=" * 56)
    print("  Migration: enrolled.csv → MySQL + FAISS")
    print("=" * 56)

    os.makedirs(FAISS_DIR, exist_ok=True)

    # ── Load CSV ─────────────────────────────────────────────────────────
    records = load_csv(ENROLLED_CSV)
    if not records:
        print("[INFO] No records in CSV — nothing to migrate.")
        return

    # ── Connect MySQL ─────────────────────────────────────────────────────
    print(f"\n[DB] Connecting to MySQL ...")
    try:
        conn   = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("[DB] Connected.\n")
    except mysql.connector.Error as e:
        print(f"[ERROR] MySQL connection failed: {e}")
        sys.exit(1)

    # ── Per-class FAISS indexes (built in memory, saved at end) ──────────
    # key = class_name, value = faiss.IndexIDMap
    class_indexes: Dict[str, faiss.IndexIDMap] = {}
    class_ids:     Dict[str, int]              = {}

    # ── Migrate each student ──────────────────────────────────────────────
    migrated  = 0
    skipped   = 0
    failed    = 0

    print(f"[MIGRATE] Processing {len(records)} student(s) ...\n")
    print("  For each student you'll be asked which class they belong to.")
    print("  Type the class name exactly as you want it stored")
    print("  (e.g. CS-A, MECH-B, 11-Science, 10-B)\n")
    print("─" * 56)

    for rec in records:
        csv_id = rec["csv_student_id"]
        name   = rec["name"]
        embs   = rec["embeddings"]

        print(f"\n  Student : {name}  (CSV id: {csv_id})")
        print(f"  Angles  : {len(embs)} embedding(s)")

        # Ask class name
        while True:
            class_name = input(
                f"  Class for {name} (e.g. CS-A) : "
            ).strip()
            if class_name:
                break
            print("  [!] Class name cannot be empty.")

        # Ask roll number
        while True:
            roll_no = input(
                f"  Roll number for {name}       : "
            ).strip()
            if roll_no:
                break
            print("  [!] Roll number cannot be empty.")

        try:
            # Get or create class in MySQL
            if class_name not in class_ids:
                class_id = get_or_create_class(
                    cursor, SCHOOL_ID, class_name)
                class_ids[class_name] = class_id
                conn.commit()
                print(f"  [DB] Class '{class_name}' → id={class_id}")
            else:
                class_id = class_ids[class_name]

            # Check if already migrated
            existing_id = student_exists(
                cursor, SCHOOL_ID, class_id, roll_no)
            if existing_id:
                print(f"  [SKIP] Already in MySQL as id={existing_id}")
                skipped += 1

                # Still add to FAISS if not there
                index_path = os.path.join(
                    FAISS_DIR, f"{class_name}.index")
                if class_name not in class_indexes:
                    class_indexes[class_name] = \
                        load_or_create_index(index_path)
                add_to_index(
                    class_indexes[class_name], embs, existing_id)
                continue

            # Photo path
            photo_path = os.path.join(
                "enrolled_images", csv_id)

            # Insert into MySQL → get permanent id
            student_id = insert_student(
                cursor, SCHOOL_ID, class_id,
                name, roll_no, embs, photo_path)
            conn.commit()
            print(f"  [DB] Inserted → MySQL id={student_id}")

            # Add to FAISS index for this class
            index_path = os.path.join(
                FAISS_DIR, f"{class_name}.index")
            if class_name not in class_indexes:
                class_indexes[class_name] = \
                    load_or_create_index(index_path)
            add_to_index(
                class_indexes[class_name], embs, student_id)
            print(f"  [FAISS] Added to {class_name}.index "
                  f"with id={student_id}")

            migrated += 1

        except Exception as e:
            print(f"  [ERROR] Failed for {name}: {e}")
            conn.rollback()
            failed += 1

    # ── Save all FAISS indexes ────────────────────────────────────────────
    print("\n[FAISS] Saving all index files ...")
    for class_name, index in class_indexes.items():
        path = os.path.join(FAISS_DIR, f"{class_name}.index")
        save_index(index, path)

    cursor.close()
    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print(f"  Migration complete")
    print(f"  Migrated : {migrated}")
    print(f"  Skipped  : {skipped}  (already in MySQL)")
    print(f"  Failed   : {failed}")
    print(f"\n  FAISS indexes saved to: {FAISS_DIR}/")
    for fname in os.listdir(FAISS_DIR):
        if fname.endswith(".index"):
            fpath = os.path.join(FAISS_DIR, fname)
            idx   = faiss.read_index(fpath)
            print(f"    {fname:<30} {idx.ntotal} vectors")
    print("=" * 56 + "\n")
    print("  Next step: run Enroll_student.py")
    print("  It now writes directly to MySQL + FAISS.\n")


if __name__ == "__main__":
    run()