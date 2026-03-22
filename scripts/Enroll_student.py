"""
Auto Enrollment Pipeline  —  MySQL + FAISS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model   : InsightFace buffalo_l  (ResNet-50 ArcFace — best CPU accuracy)
Storage : MySQL  →  students table  (embedding BLOBs)
          FAISS  →  faiss_indexes/<class_name>.index  (per-class IDMap)

Enrollment flow:
  1. Face detected → quality gate
  2. 3-angle capture → 3 × 512-dim embeddings
  3. FAISS duplicate check across ALL classes
  4. If new person:
       a. Ask class, roll number, name
       b. MySQL INSERT students  →  returns permanent student_id
       c. FAISS add_with_ids(embeddings, student_id)  <- same id forever
       d. Save .index file to disk
  5. If already enrolled → show name, do not save

FAISS design:
  - One IndexIDMap per class  (e.g. CS-A.index, MECH-B.index)
  - IDs in FAISS = MySQL students.id  (permanent, never changes)
  - Rebuilt from MySQL embedding BLOBs any time  (MySQL is source of truth)
  - L2-normalised before add  ->  inner product = cosine similarity

Install:
  pip install opencv-python==4.8.1.78 numpy==1.26.4 insightface onnxruntime
  pip install mysql-connector-python faiss-cpu

Prerequisites:
  Run db_setup.py first to create MySQL schema.
  Run migrate_csv.py if you have existing enrolled.csv data.
"""

import cv2
import os
import sys
import time
import warnings
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Protocol, runtime_checkable

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
#  FACE PROTOCOL
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class FaceObject(Protocol):
    bbox:      np.ndarray
    det_score: float
    embedding: Optional[np.ndarray]
    pose:      Optional[np.ndarray]


try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("[ERROR] Run: pip install insightface onnxruntime")
    sys.exit(1)

try:
    import mysql.connector
    from mysql.connector import MySQLConnection
except ImportError:
    print("[ERROR] Run: pip install mysql-connector-python")
    sys.exit(1)

try:
    import faiss
except ImportError:
    print("[ERROR] Run: pip install faiss-cpu")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
from config import db_password
CAMERA_INDEX    = 0
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
IMAGES_DIR      = "enrolled_images"
FAISS_DIR       = "faiss_indexes"
MODELS_DIR      = "models"   # InsightFace models stored here inside project

# ── MySQL ─────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": db_password,   # <- change to your MySQL password
    "database": "attendance_system",
}
SCHOOL_ID = 1

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME    = "buffalo_sc"
EMBEDDING_DIM = 512

# ── Tuning ────────────────────────────────────────────────────────────────────
CAPTURE_INTERVAL = 5.0
DET_SCORE_MIN    = 0.60
NORM_MIN         = 18.0   # lowered from 20 — buffalo_sc avg light gives 18-22
                           # still blocks hidden faces (norm 8-15)
YAW_MAX          = 35.0

ANGLES_NEEDED = 3
ANGLE_PROMPTS = [
    "Look straight at camera",
    "Slightly turn LEFT",
    "Slightly turn RIGHT",
]


# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def emb_to_blob(emb: np.ndarray) -> bytes:
    return emb.astype(np.float32).tobytes()


def blob_to_emb(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def l2_norm(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb)
    return emb / n if n > 0 else emb


# ─────────────────────────────────────────────────────────────────────────────
#  MYSQL LAYER
# ─────────────────────────────────────────────────────────────────────────────

def db_connect() -> MySQLConnection:
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as e:
        print(f"\n[ERROR] MySQL connection failed: {e}")
        print("  Check DB_CONFIG — host, port, user, password, database.")
        sys.exit(1)


def db_get_or_create_class(conn: MySQLConnection,
                            class_name: str) -> int:
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM classes "
        "WHERE school_id=%s AND name=%s LIMIT 1",
        (SCHOOL_ID, class_name),
    )
    row = cur.fetchone()
    if row:
        cur.close()
        return int(row[0])
    cur.execute(
        "INSERT INTO classes (school_id, name) VALUES (%s, %s)",
        (SCHOOL_ID, class_name),
    )
    conn.commit()
    class_id = cur.lastrowid
    cur.close()
    print(f"[DB] New class: '{class_name}' id={class_id}")
    return class_id


def db_roll_exists(conn: MySQLConnection,
                   class_id: int, roll_no: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM students "
        "WHERE school_id=%s AND class_id=%s AND roll_no=%s LIMIT 1",
        (SCHOOL_ID, class_id, roll_no),
    )
    exists = cur.fetchone() is not None
    cur.close()
    return exists


def db_insert_student(conn: MySQLConnection,
                      class_id: int,
                      name: str,
                      roll_no: str,
                      embeddings: List[np.ndarray],
                      photo_path: str) -> int:
    blobs = [emb_to_blob(e) for e in embeddings]
    while len(blobs) < 3:
        blobs.append(None)
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO students
           (school_id, class_id, name, roll_no,
            emb_1, emb_2, emb_3, photo_path)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        (SCHOOL_ID, class_id, name, roll_no,
         blobs[0], blobs[1], blobs[2], photo_path),
    )
    conn.commit()
    student_id = cur.lastrowid
    cur.close()
    return student_id


def db_load_all_students(conn: MySQLConnection) -> List[Dict]:
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """SELECT s.id AS student_id, s.name, s.class_id,
                  s.emb_1, s.emb_2, s.emb_3,
                  c.name AS class_name
           FROM   students s
           JOIN   classes  c ON c.id = s.class_id
           WHERE  s.school_id = %s""",
        (SCHOOL_ID,),
    )
    rows = cur.fetchall()
    cur.close()
    records = []
    for row in rows:
        embs = []
        for key in ("emb_1", "emb_2", "emb_3"):
            if row[key]:
                embs.append(blob_to_emb(row[key]))
        if embs:
            records.append({
                "student_id": int(row["student_id"]),
                "name":       str(row["name"]),
                "class_id":   int(row["class_id"]),
                "class_name": str(row["class_name"]),
                "embeddings": embs,
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
#  FAISS LAYER
# ─────────────────────────────────────────────────────────────────────────────

def faiss_index_path(class_name: str, school_name: str = "Default School") -> str:
    school_dir = os.path.join(FAISS_DIR, school_name)
    os.makedirs(school_dir, exist_ok=True)
    return os.path.join(school_dir, f"{class_name}.index")


def faiss_new_index() -> faiss.IndexIDMap:
    base = faiss.IndexFlatIP(EMBEDDING_DIM)
    return faiss.IndexIDMap(base)


def faiss_add(index: faiss.IndexIDMap,
              embeddings: List[np.ndarray],
              student_id: int) -> None:
    for emb in embeddings:
        vec = l2_norm(emb).reshape(1, -1).astype(np.float32)
        ids = np.array([student_id], dtype=np.int64)
        index.add_with_ids(vec, ids)


def faiss_save(index: faiss.IndexIDMap, class_name: str) -> None:
    faiss.write_index(index, faiss_index_path(class_name))


def faiss_search_aggregated(
    live_emb: np.ndarray,
    indexes: Dict[str, faiss.IndexIDMap],
) -> Dict[int, List[float]]:
    """
    Search ALL vectors in ALL class indexes.
    Returns dict: { student_id: [similarity_score, ...] }

    Uses k=ntotal to retrieve every stored vector's score,
    then groups scores by student_id for aggregation.
    This prevents a single lucky side-face match from winning
    — the student must score well across ALL their stored angles.
    """
    vec            = l2_norm(live_emb).reshape(1, -1).astype(np.float32)
    id_scores: Dict[int, List[float]] = {}

    for index in indexes.values():
        if index.ntotal == 0:
            continue
        k = index.ntotal   # retrieve scores against every stored vector
        scores, ids = index.search(vec, k)
        for score, sid in zip(scores[0], ids[0]):
            sid = int(sid)
            if sid == -1:
                continue
            similarity = float(score)   # inner product on L2-normed = cosine sim
            if sid not in id_scores:
                id_scores[sid] = []
            id_scores[sid].append(similarity)

    return id_scores


def aggregate_scores(
    id_scores: Dict[int, List[float]],
) -> Dict[int, float]:
    """
    Compute average cosine similarity per student across all their vectors.
    Using average (not sum) makes it fair when students have different
    numbers of stored embeddings (e.g. 1 vs 3 angles).

    Returns dict: { student_id: avg_similarity }
    """
    return {
        sid: sum(scores) / len(scores)
        for sid, scores in id_scores.items()
        if scores
    }


def build_indexes_from_mysql(
    students: List[Dict],
) -> Dict[str, faiss.IndexIDMap]:
    """Build fresh in-memory FAISS indexes from MySQL student records."""
    indexes: Dict[str, faiss.IndexIDMap] = {}
    for rec in students:
        cname = rec["class_name"]
        if cname not in indexes:
            indexes[cname] = faiss_new_index()
        faiss_add(indexes[cname], rec["embeddings"], rec["student_id"])
    return indexes


# ─────────────────────────────────────────────────────────────────────────────
#  MATCHING
# ─────────────────────────────────────────────────────────────────────────────

# Similarity threshold — cosine SIMILARITY (higher = more similar, 0.0 to 1.0)
# buffalo_sc sweet spot: 0.45  (your test showed sim=0.48 for half-hidden face)
# buffalo_l sweet spot: 0.55
# AdaFace production:   0.60
SIMILARITY_THRESHOLD = 0.40


def find_match_faiss(
    live_embeddings: List[np.ndarray],
    indexes: Dict[str, faiss.IndexIDMap],
    id_to_record: Dict[int, Dict],
) -> Tuple[bool, int, str, float, Dict[int, float]]:
    """
    Aggregated score matching across all live embeddings and all stored vectors.

    For each live embedding:
      - Search all stored vectors across all class indexes
      - Collect per-student similarity scores

    Then aggregate across all live embeddings:
      - Per student: average similarity across ALL live × stored comparisons
      - Winner = student with highest average similarity
      - Only accept if winner_avg >= SIMILARITY_THRESHOLD

    This prevents hallucination from side-face coincidence:
      A side face might score 0.81 against one stored vector of the wrong person,
      but that person's OTHER stored vectors will score low (0.3-0.4),
      pulling their average down below the correct person's average.

    Returns (matched, student_id, name, avg_similarity, all_avg_scores).
    """
    # Accumulate scores: { student_id: [sim1, sim2, ...] }
    # One list entry per (live_embedding × stored_vector) comparison
    combined: Dict[int, List[float]] = {}

    for live_emb in live_embeddings:
        id_scores = faiss_search_aggregated(live_emb, indexes)
        for sid, scores in id_scores.items():
            if sid not in combined:
                combined[sid] = []
            combined[sid].extend(scores)

    if not combined:
        return False, -1, "", 0.0, {}

    # Average similarity per student across all comparisons
    avg_scores = aggregate_scores(combined)

    # Winner = highest average similarity
    best_id  = max(avg_scores, key=lambda s: avg_scores[s])
    best_avg = avg_scores[best_id]
    best_name = id_to_record.get(best_id, {}).get("name", "Unknown")

    matched = best_avg >= SIMILARITY_THRESHOLD

    return matched, best_id, best_name, best_avg, avg_scores


# ─────────────────────────────────────────────────────────────────────────────
#  LIGHTING + BLUR CHECK  (frame-level, before face quality)
# ─────────────────────────────────────────────────────────────────────────────

# Thresholds — tune these for your classroom environment
BRIGHTNESS_MIN  = 60    # mean pixel value below this = too dark
BRIGHTNESS_MAX  = 210   # mean pixel value above this = too bright / overexposed
CONTRAST_MIN    = 30    # std deviation below this = flat / washed out
BLUR_MIN        = 80.0  # Laplacian variance below this = blurry frame

# Face size thresholds (pixels)
FACE_SIZE_MIN   = 80    # face bbox shorter side below this = too far
FACE_SIZE_MAX   = 400   # face bbox shorter side above this = too close

# Pose thresholds (degrees)
PITCH_MAX       = 30.0  # looking up or down too much
ROLL_MAX        = 30.0  # head tilted sideways too much


def check_frame_lighting(frame: np.ndarray) -> Tuple[bool, str]:
    """
    Check overall frame lighting using the face region.
    Returns (ok, reason).
    Reason is empty string when ok=True.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    std  = float(np.std(gray))

    if mean < BRIGHTNESS_MIN:
        return False, f"Too dark ({mean:.0f}) — turn on more light"
    if mean > BRIGHTNESS_MAX:
        return False, f"Too bright ({mean:.0f}) — reduce glare"
    if std < CONTRAST_MIN:
        return False, f"Low contrast ({std:.0f}) — check lighting"
    return True, ""


def check_frame_blur(frame: np.ndarray) -> Tuple[bool, str]:
    """
    Laplacian variance blur detection.
    Lower variance = blurrier frame.
    Returns (ok, reason).
    """
    gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if variance < BLUR_MIN:
        return False, f"Blurry ({variance:.0f}) — hold still"
    return True, ""


def check_face_lighting(face_crop: np.ndarray) -> Tuple[bool, str]:
    """
    Check lighting specifically on the face crop.
    Catches cases where the overall frame is bright but the face is shadowed.
    """
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    std  = float(np.std(gray))
    if mean < BRIGHTNESS_MIN:
        return False, f"Face too dark ({mean:.0f}) — face the light"
    if mean > BRIGHTNESS_MAX:
        return False, f"Face overexposed ({mean:.0f}) — avoid direct light"
    if std < CONTRAST_MIN:
        return False, f"Face poorly lit ({std:.0f}) — even lighting needed"
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
#  FACE QUALITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

def quality_score(face: FaceObject,
                  frame: Optional[np.ndarray] = None) -> Tuple[float, str]:
    """
    Comprehensive quality score covering:
      - Detection confidence
      - Embedding norm (occlusion proxy)
      - Head pose: yaw, pitch, roll
      - Face size (too far / too close)
      - Frame lighting (dark, bright, low contrast)
      - Frame blur
      - Face crop lighting

    Returns (score_0_to_100, reason_string).
    reason is empty when score is high and no hard blocks triggered.
    """
    try:
        det  = float(face.det_score)
        emb  = face.embedding
        norm = float(np.linalg.norm(emb)) if emb is not None else 0.0
        pose = face.pose  # [pitch, yaw, roll] in degrees
        yaw   = abs(float(pose[1])) if pose is not None and len(pose) > 1 else 0.0
        pitch = abs(float(pose[0])) if pose is not None and len(pose) > 0 else 0.0
        roll  = abs(float(pose[2])) if pose is not None and len(pose) > 2 else 0.0
    except Exception as e:
        return 0.0, f"Read error: {e}"

    # ── Hard blocks — fail immediately with clear message ─────────────────

    # 1. Detection confidence
    if det < DET_SCORE_MIN:
        return float(int(det * 1000) / 10), \
               f"Low confidence ({det:.2f}) — move closer"

    # 2. Embedding norm — face likely occluded or partially hidden
    if norm < NORM_MIN:
        return float(int((norm / NORM_MIN) * 500) / 10), \
               f"Face obscured (norm={norm:.1f}) — remove obstructions"

    # 3. Face size — too far away
    try:
        b  = face.bbox
        bw = int(b[2]) - int(b[0])
        bh = int(b[3]) - int(b[1])
        shorter_side = min(bw, bh)
        if shorter_side < FACE_SIZE_MIN:
            return 20.0, f"Too far away ({shorter_side}px) — move closer"
        if shorter_side > FACE_SIZE_MAX:
            return 20.0, f"Too close ({shorter_side}px) — move back"
    except Exception:
        pass

    # 4. Yaw — head turned left/right too much
    if yaw > YAW_MAX:
        s = max(0.0, 100.0 - yaw * 2)
        return float(int(s * 10) / 10), \
               f"Turn head forward ({yaw:.0f}deg yaw)"

    # 5. Pitch — looking up or down too much
    if pitch > PITCH_MAX:
        s = max(0.0, 100.0 - pitch * 2)
        return float(int(s * 10) / 10), \
               f"Look straight ahead ({pitch:.0f}deg pitch)"

    # 6. Roll — head tilted sideways too much
    if roll > ROLL_MAX:
        s = max(0.0, 100.0 - roll * 2)
        return float(int(s * 10) / 10), \
               f"Straighten head ({roll:.0f}deg roll)"

    # 7. Frame-level lighting check
    if frame is not None:
        light_ok, light_reason = check_frame_lighting(frame)
        if not light_ok:
            return 15.0, light_reason

        blur_ok, blur_reason = check_frame_blur(frame)
        if not blur_ok:
            return 25.0, blur_reason

        # 8. Face crop lighting check
        try:
            h, w = frame.shape[:2]
            b    = face.bbox
            x1   = max(0, int(b[0]))
            y1   = max(0, int(b[1]))
            x2   = min(w, int(b[2]))
            y2   = min(h, int(b[3]))
            if x2 > x1 and y2 > y1:
                face_crop = frame[y1:y2, x1:x2]
                face_ok, face_reason = check_face_lighting(face_crop)
                if not face_ok:
                    return 15.0, face_reason
        except Exception:
            pass

    # ── Soft scoring — all checks passed, compute composite score ─────────
    det_s   = min(det, 1.0)
    norm_s  = min(norm / 25.0, 1.0)
    yaw_s   = max(0.0, 1.0 - yaw   / YAW_MAX)
    pitch_s = max(0.0, 1.0 - pitch / PITCH_MAX)
    roll_s  = max(0.0, 1.0 - roll  / ROLL_MAX)

    score = (det_s   * 0.25
           + norm_s  * 0.40
           + yaw_s   * 0.15
           + pitch_s * 0.10
           + roll_s  * 0.10) * 100

    return float(int(score * 10) / 10), ""


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_bbox(face: FaceObject) -> Tuple[int, int, int, int]:
    try:
        b = face.bbox
        return int(b[0]), int(b[1]), int(b[2]), int(b[3])
    except Exception:
        return 0, 0, 0, 0


def crop_face(frame: np.ndarray, face: FaceObject,
              pad: int = 20) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = get_bbox(face)
    return frame[
        max(0, y1 - pad): min(h, y2 + pad),
        max(0, x1 - pad): min(w, x2 + pad)
    ].copy()


def get_best_face(
    faces: list,
    frame: Optional[np.ndarray] = None,
) -> Tuple[Optional[FaceObject], float, str]:
    best_face:   Optional[FaceObject] = None
    best_score:  float = 0.0
    best_reason: str   = "No face detected"
    for face in faces:
        f: FaceObject = face
        s, r = quality_score(f, frame)
        if s > best_score:
            best_score  = s
            best_reason = r
            best_face   = f
    return best_face, best_score, best_reason


def save_face_images(student_id: int,
                     crops: List[np.ndarray]) -> str:
    d = os.path.join(IMAGES_DIR, str(student_id))
    os.makedirs(d, exist_ok=True)
    for i, crop in enumerate(crops):
        cv2.imwrite(os.path.join(d, f"{i+1}.jpg"), crop)
    print(f"[IMG] {len(crops)} images → {d}/")
    return d


# ─────────────────────────────────────────────────────────────────────────────
#  HUD
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(frame: np.ndarray, score: float, reason: str,
             countdown: float, n_enrolled: int,
             msg: str, msg_color: tuple) -> None:
    h, w    = frame.shape[:2]
    score_f = float(score)

    # ── Quality bar ───────────────────────────────────────────────────────
    cv2.rectangle(frame, (10, 10), (310, 40), (40, 40, 40), -1)
    bw = int((min(score_f, 100.0) / 100.0) * 298)
    bc = ((0, 200, 80)  if score_f >= 75
          else (0, 180, 220) if score_f >= 50
          else (40, 60, 220))
    if bw > 0:
        cv2.rectangle(frame, (11, 11), (11 + bw, 39), bc, -1)
    cv2.putText(frame, f"Quality: {score_f:.0f}%", (15, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

    # ── Lighting indicator (top right) ────────────────────────────────────
    gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness  = float(np.mean(gray))
    light_ok    = BRIGHTNESS_MIN <= brightness <= BRIGHTNESS_MAX
    light_color = (0, 200, 80) if light_ok else (40, 60, 220)
    light_label = f"Light: {brightness:.0f}"
    cv2.putText(frame, light_label, (w - 130, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                light_color, 1, cv2.LINE_AA)
    if not light_ok:
        warn = "Too dark" if brightness < BRIGHTNESS_MIN else "Too bright"
        cv2.putText(frame, warn, (w - 130, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (40, 60, 220), 1, cv2.LINE_AA)

    # ── Reason and message ────────────────────────────────────────────────
    if reason:
        cv2.putText(frame, reason, (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (60, 100, 230), 1, cv2.LINE_AA)
    if msg:
        cv2.putText(frame, msg, (10, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    msg_color, 2, cv2.LINE_AA)

    # ── Countdown bar ─────────────────────────────────────────────────────
    filled = int((1.0 - countdown / CAPTURE_INTERVAL) * (w - 40))
    cv2.rectangle(frame, (20, h - 20), (w - 20, h - 8),
                  (50, 50, 50), -1)
    if filled > 0:
        cv2.rectangle(frame, (20, h - 20),
                      (20 + filled, h - 8), (255, 180, 0), -1)
    cv2.putText(frame, f"Scan in: {countdown:.1f}s", (10, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Enrolled: {n_enrolled}", (w - 160, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, "MySQL+FAISS", (w - 120, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (100, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "S=Stop  Q=Quit", (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (100, 220, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  3-ANGLE CAPTURE  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def capture_three_angles(
    app: object,
    cap: object,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    embs:  List[np.ndarray] = []
    crops: List[np.ndarray] = []

    for i in range(ANGLES_NEEDED):
        prompt = (ANGLE_PROMPTS[i]
                  if i < len(ANGLE_PROMPTS) else f"Angle {i+1}")
        print(f"\n  [{i+1}/{ANGLES_NEEDED}] {prompt}")
        print("  SPACE = capture   Q = cancel")

        while True:
            ret   = False
            frame = None
            if hasattr(cap, "read"):
                ret, frame = cap.read()  # type: ignore
            if not ret or frame is None:
                continue

            display = frame.copy()

            cv2.rectangle(display, (0, 0), (FRAME_WIDTH, 100),
                          (15, 15, 60), -1)
            cv2.putText(display,
                        f"[{i+1}/{ANGLES_NEEDED}]  {prompt}",
                        (10, 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display,
                        "SPACE = capture this angle   Q = cancel",
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (180, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(display,
                        "Each angle stored separately — best accuracy",
                        (10, 82), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (120, 180, 120), 1, cv2.LINE_AA)

            for j in range(ANGLES_NEEDED):
                col = (0, 220, 80) if j < i else (70, 70, 70)
                cx  = FRAME_WIDTH - 25 - (ANGLES_NEEDED - 1 - j) * 26
                cv2.circle(display, (cx, 26), 9, col, -1)

            try:
                faces = app.get(frame)  # type: ignore
            except Exception:
                faces = []

            best_f, best_s, best_r = get_best_face(faces, frame)

            if best_f is not None:
                x1, y1, x2, y2 = get_bbox(best_f)
                col = ((0, 210, 80) if float(best_s) >= 75.0
                       else (40, 60, 220))
                cv2.rectangle(display, (x1, y1), (x2, y2), col, 2)
                cv2.putText(display, f"{best_s:.0f}%",
                            (x1, max(y1 - 8, 110)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            col, 1, cv2.LINE_AA)
                if best_f.embedding is not None:
                    n = float(np.linalg.norm(best_f.embedding))
                    cv2.putText(display, f"norm={n:.1f}",
                                (x1, max(y1 - 24, 110)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                                (180, 180, 180), 1, cv2.LINE_AA)
            else:
                cv2.putText(display, best_r, (10, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                            (60, 80, 200), 1, cv2.LINE_AA)

            cv2.imshow("Auto Enrollment", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if best_f is None or float(best_s) < 75.0:
                    print(f"  [!] Quality {best_s:.0f}% too low"
                          f" — adjust and retry")
                    continue
                emb = best_f.embedding
                if emb is None:
                    print("  [!] No embedding — retry")
                    continue
                # Hard norm check on the ACTUAL captured embedding
                # This is the definitive check — preview score can lag
                # Healthy full face: 22-28
                # Partially hidden:  8-17  (hand, mask, hair)
                # Edge cases:        18-21 (dim light, slight occlusion)
                norm = float(np.linalg.norm(emb))
                if norm < NORM_MIN:
                    print(f"  [!] Embedding quality too low "
                          f"(norm={norm:.1f} < {NORM_MIN}) "
                          f"— show full face in good lighting")
                    continue
                embs.append(emb.copy())
                crops.append(crop_face(frame, best_f))
                print(f"  [✓] Angle {i+1} captured  "
                      f"quality={best_s:.0f}%  "
                      f"norm={norm:.1f}")
                break

            elif key in (ord("q"), ord("Q")):
                print("  [CANCEL] Capture cancelled.")
                return [], []

    return embs, crops


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(FAISS_DIR,  exist_ok=True)

    # ── Connect MySQL ─────────────────────────────────────────────────────
    print("[DB] Connecting to MySQL ...")
    conn = db_connect()
    print("[DB] Connected.\n")

    # ── Load all students → build FAISS indexes in RAM ────────────────────
    print("[DB] Loading enrolled students from MySQL ...")
    all_students = db_load_all_students(conn)
    print(f"[DB] {len(all_students)} student(s) loaded.\n")

    id_to_record: Dict[int, Dict] = {
        r["student_id"]: r for r in all_students
    }

    print("[FAISS] Building indexes from MySQL embeddings ...")
    class_indexes = build_indexes_from_mysql(all_students)
    total_vecs    = sum(idx.ntotal for idx in class_indexes.values())
    print(f"[FAISS] {len(class_indexes)} class index(es)  "
          f"{total_vecs} total vectors.\n")

    # Persist fresh indexes to disk
    for cname, idx in class_indexes.items():
        faiss_save(idx, cname)

    enrolled_count = len(all_students)

    # ── InsightFace model ─────────────────────────────────────────────────
    # InsightFace automatically appends a 'models' subfolder to root.
    # So we pass the project root directly — it saves to project/models/buffalo_sc/
    models_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    print(f"[InsightFace] Loading {MODEL_NAME} ...")
    print(f"[InsightFace] Model path: {os.path.join(models_root, 'models', MODEL_NAME)}")
    app = FaceAnalysis(
        name=MODEL_NAME,
        root=models_root,
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Suppress InsightFace FutureWarnings that fire lazily on first face process
    # These are internal to InsightFace (rcond + SimilarityTransform deprecations)
    # and do not affect correctness in any way
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning,
                            module="insightface")

    print(f"[InsightFace] {MODEL_NAME} ready.\n")

    # ── Camera ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Try CAMERA_INDEX = 1")
        conn.close()
        return

    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  AUTO ENROLLMENT  —  MySQL + FAISS + {MODEL_NAME:<14}║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Face detected → 3 angle captures (SPACE each)     ║")
    print("║  Duplicate check: FAISS search across all classes   ║")
    print("║  New student → MySQL INSERT → FAISS add_with_ids   ║")
    print("║  MySQL id = FAISS id  (permanent, never changes)    ║")
    print("║  S = stop   Q = quit                                ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    last_scan:  float = time.time() - CAPTURE_INTERVAL
    msg:        str   = "Ready — waiting for face..."
    msg_color:  tuple = (200, 200, 200)
    cur_score:  float = 0.0
    cur_reason: str   = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        display   = frame.copy()
        now       = time.time()
        countdown = max(0.0, CAPTURE_INTERVAL - (now - last_scan))

        try:
            faces = app.get(frame)
        except Exception as e:
            faces = []
            print(f"[WARN] {e}")

        best_face, best_score, best_reason = get_best_face(faces, frame)

        if best_face is not None:
            x1, y1, x2, y2 = get_bbox(best_face)
            bc = ((0, 210, 80) if float(best_score) >= 75.0
                  else (40, 60, 220))
            cv2.rectangle(display, (x1, y1), (x2, y2), bc, 2)
            cv2.putText(display, f"{best_score:.0f}%",
                        (x1, max(y1 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        bc, 1, cv2.LINE_AA)

        cur_score  = best_score
        cur_reason = best_reason if best_reason else ""

        # ── Scan trigger ─────────────────────────────────────────────────
        if countdown <= 0.0:

            if best_face is None:
                msg       = "No face — stand in front of camera"
                msg_color = (40, 80, 220)
                last_scan = time.time()

            elif float(best_score) < 75.0:
                msg       = (f"Quality low ({best_score:.0f}%)"
                             f" — {cur_reason}")
                msg_color = (40, 80, 220)
                last_scan = time.time()

            else:
                # ── Step 1: capture 3 angles ──────────────────────────────
                print("\n[SCAN] Face detected — capturing 3 angles...")

                banner = display.copy()
                cv2.rectangle(banner, (0, 0), (FRAME_WIDTH, 90),
                              (50, 35, 0), -1)
                cv2.putText(banner,
                            "Face detected — follow prompts below",
                            (10, 36), cv2.FONT_HERSHEY_SIMPLEX,
                            0.72, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(banner,
                            "Press SPACE for each of the 3 angles",
                            (10, 66), cv2.FONT_HERSHEY_SIMPLEX,
                            0.46, (255, 210, 120), 1, cv2.LINE_AA)
                cv2.imshow("Auto Enrollment", banner)
                cv2.waitKey(1200)

                new_embs, crops = capture_three_angles(app, cap)
                last_scan = time.time()

                if len(new_embs) < ANGLES_NEEDED:
                    msg       = "Capture cancelled — waiting..."
                    msg_color = (120, 120, 120)

                else:
                    # ── Step 2: FAISS aggregated duplicate check ──────────
                    matched, sid, name, avg_sim, all_scores = \
                        find_match_faiss(
                            new_embs, class_indexes, id_to_record)

                    print(f"[MATCH] winner_avg_sim={avg_sim:.4f}  "
                          f"threshold={SIMILARITY_THRESHOLD}  "
                          f"matched={matched}")
                    if all_scores:
                        print(f"[SCORES] per student avg similarities:")
                        for s_id, s_avg in sorted(
                                all_scores.items(),
                                key=lambda x: x[1], reverse=True):
                            sname = id_to_record.get(
                                s_id, {}).get("name", f"id={s_id}")
                            print(f"  {sname:<25} avg={s_avg:.4f}"
                                  f"{'  ← winner' if s_id == sid else ''}")

                    if matched:
                        msg       = (f"Already enrolled: {name}"
                                     f"  sim={avg_sim:.3f}")
                        msg_color = (0, 160, 255)
                        print(f"[RESULT] Already enrolled → "
                              f"{name} (id={sid})\n")

                    else:
                        # ── Step 3: new — ask details ─────────────────────
                        print(f"[RESULT] New person confirmed "
                              f"(best_avg_sim={avg_sim:.4f})\n")

                        freeze = display.copy()
                        cv2.rectangle(freeze, (0, 0),
                                      (FRAME_WIDTH, 90), (0, 80, 0), -1)
                        cv2.putText(freeze, "New student confirmed!",
                                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.78, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(freeze,
                                    "Enter details in terminal",
                                    (10, 66), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.47, (200, 255, 200), 1, cv2.LINE_AA)
                        cv2.imshow("Auto Enrollment", freeze)
                        cv2.waitKey(1)

                        print("─" * 52)

                        # Class
                        while True:
                            class_name = input(
                                "  Class      (e.g. CS-A)  : "
                            ).strip()
                            if class_name:
                                break
                            print("  [!] Cannot be empty.")

                        class_id = db_get_or_create_class(conn, class_name)

                        # Roll number
                        while True:
                            roll_no = input(
                                "  Roll No    (e.g. ST001) : "
                            ).strip()
                            if not roll_no:
                                print("  [!] Cannot be empty.")
                                continue
                            if db_roll_exists(conn, class_id, roll_no):
                                print(f"  [!] Roll '{roll_no}' already"
                                      f" exists in {class_name}.")
                                continue
                            break

                        # Name
                        while True:
                            new_name = input(
                                "  Full Name              : "
                            ).strip()
                            if new_name:
                                break
                            print("  [!] Cannot be empty.")

                        # ── Step 4a: MySQL INSERT ─────────────────────────
                        student_id = db_insert_student(
                            conn, class_id, new_name, roll_no,
                            new_embs, "pending")
                        print(f"[DB] Inserted → student_id={student_id}")

                        # Save images using MySQL id as folder name
                        save_face_images(student_id, crops)
                        photo_path = os.path.join(
                            IMAGES_DIR, str(student_id))

                        # Update photo_path now we have the permanent id
                        cur = conn.cursor()
                        cur.execute(
                            "UPDATE students SET photo_path=%s WHERE id=%s",
                            (photo_path, student_id),
                        )
                        conn.commit()
                        cur.close()

                        # ── Step 4b: FAISS add_with_ids ───────────────────
                        if class_name not in class_indexes:
                            class_indexes[class_name] = faiss_new_index()
                        faiss_add(
                            class_indexes[class_name],
                            new_embs, student_id)
                        faiss_save(class_indexes[class_name], class_name)

                        n_vecs = class_indexes[class_name].ntotal
                        print(f"[FAISS] id={student_id} → "
                              f"{class_name}.index "
                              f"({n_vecs} total vectors)")

                        # Update in-memory state
                        id_to_record[student_id] = {
                            "student_id": student_id,
                            "name":       new_name,
                            "class_id":   class_id,
                            "class_name": class_name,
                            "embeddings": new_embs,
                        }
                        enrolled_count += 1

                        ts = datetime.now().strftime("%H:%M:%S")
                        print(f"\n[✓ ENROLLED]  {ts}  |  {new_name}"
                              f"  class={class_name}  roll={roll_no}"
                              f"  MySQL/FAISS id={student_id}\n")

                        msg       = (f"Enrolled: {new_name} "
                                     f"({class_name} / {roll_no})")
                        msg_color = (0, 220, 80)
                        last_scan = time.time()

        draw_hud(display, cur_score, cur_reason,
                 countdown, enrolled_count, msg, msg_color)

        cv2.imshow("Auto Enrollment", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("s"), ord("S")):
            print("\n[STOP] Stopped.")
            break
        elif key in (ord("q"), ord("Q")):
            print("\n[QUIT] Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

    print("\n" + "═" * 52)
    print(f"  DONE — {enrolled_count} student(s) in MySQL")
    print(f"  FAISS indexes → {FAISS_DIR}/")
    for fname in sorted(os.listdir(FAISS_DIR)):
        if fname.endswith(".index"):
            idx = faiss.read_index(os.path.join(FAISS_DIR, fname))
            print(f"    {fname:<30} {idx.ntotal} vectors")
    print("═" * 52 + "\n")


if __name__ == "__main__":
    run()