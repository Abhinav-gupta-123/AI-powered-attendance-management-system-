<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=FaceAttend%20AI&fontSize=60&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Next-Gen%20Attendance%20%7C%20Powered%20by%20Computer%20Vision&descAlignY=60&descSize=18" width="100%"/>

<p align="center">
  <img src="https://img.shields.io/badge/Status-🔥%20Actively%20Developing-brightgreen?style=for-the-badge&logo=github"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Uvicorn-ASGI%20Server-4B0082?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/InsightFace-Buffalo__SC%2FL-00FFFF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-FF6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/MySQL-Connection%20Pool-4479A1?style=for-the-badge&logo=mysql&logoColor=white"/>
  <img src="https://img.shields.io/badge/JWT-Auth%20%2B%20Roles-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/OpenCV-CV%20Processing-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
</p>

<br/>

> ### 🚀 *"Walk in. Get recognized. Attendance marked. That simple."*
> An intelligent, fully automated attendance system that uses **real-time face detection & recognition** to mark attendance — no ID cards, no roll calls, no nonsense.

<br/>

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/powered-by-coffee.svg)](https://forthebadge.com)

</div>

---

## ⚡ What Is This?

**FaceAttend AI** is a production-grade, fully automated attendance management system that uses **computer vision + deep learning** to identify people from a live camera feed and record their attendance in real time — with high accuracy, role-based access control, and multi-school support.

No manual entries. No buddy punching. No spreadsheets.

Just walk in front of the camera → recognized → ✅ **Done.**

---

## 🧠 How It Works (Actual Pipeline)

```
📸 Camera Feed
      ↓
🎯 InsightFace Buffalo_SC  ──  Face Detection + Alignment (640×640)
      ↓
🧬 InsightFace Buffalo_L   ──  512-D Face Embedding (ONNX / CPU)
      ↓
⚡ FAISS Index             ──  Per-school cosine similarity search
      ↓
🆔 Identity Matched        ──  Confidence score logged
      ↓
🗄️  MySQL (pooled)         ──  Attendance record written (1 per student/day)
      ↓
🌐 FastAPI + Uvicorn       ──  REST API + HTML Dashboard
      ↓
🔐 JWT Auth                ──  Admin / Teacher role-gated routes
```

---

## 🔥 Features

| Feature | Status |
|---|---|
| 🎯 Real-time face detection via InsightFace Buffalo_SC | ✅ Live |
| 🧬 512-D face embedding via InsightFace Buffalo_L | ✅ Live |
| ⚡ FAISS cosine similarity search (per-school index) | ✅ Live |
| 🏫 Multi-school architecture with isolated FAISS indexes | ✅ Live |
| 🗄️ MySQL with connection pooling (pool size 5) | ✅ Live |
| 🔐 JWT authentication (bcrypt + jose) | ✅ Live |
| 👤 Admin & Teacher roles with class-level access control | ✅ Live |
| 📋 Session-based attendance (one session per class/date) | ✅ Live |
| 📸 Student enrollment via webcam (3-angle embeddings) | ✅ Live |
| 🔍 FAISS index inspector UI | ✅ Live |
| 🌐 FastAPI REST API + HTML frontend dashboard | ✅ Live |
| 📊 Analytics & attendance reports | 🔄 In Progress |
| 🚫 Anti-spoofing / liveness detection | 🔄 In Progress |
| 📱 Mobile-friendly UI | 🔄 In Progress |
| 🔔 Email/SMS alerts on absence | 🧪 Planned |
| ☁️ Docker + cloud deployment | 🧪 Planned |

---

## 🏗️ Project Structure

```
AI-powered-attendance-management-system/
│
├── 📁 frontend/                  # HTML/CSS/JS Web Dashboard
│
├── 📁 models/                    # InsightFace model weights (Buffalo_SC / Buffalo_L)
│
├── 📁 scripts/
│   ├── api.py                    # FastAPI entry point — mounts all routers
│   ├── auth_deps.py              # JWT auth dependencies (get_current_user, require_admin)
│   ├── config.py                 # DB credentials
│   ├── db.py                     # MySQL connection pool + query helpers
│   ├── DB_setup.py               # One-time DB + table creation
│   ├── live_attendance.py        # Core live attendance session logic
│   ├── Enroll_student.py         # Student enrollment + embedding generation
│   ├── inspect_index.py          # FAISS index inspection utilities
│   ├── migrate_csv.py            # CSV → MySQL data migration
│   ├── migrate_auth.py           # Phase 2: users + teacher_classes tables
│   ├── migrate_faiss.py          # Move flat indexes → per-school subdirectories
│   └── routers/
│       ├── attendance.py         # Live session routes + dashboard UI
│       ├── enroll.py             # Enrollment routes + UI
│       ├── inspect.py            # FAISS inspector routes + UI
│       └── auth.py               # Login / logout / token routes
│
├── 📁 faiss_indexes/
│   └── Default School/           # Per-school FAISS index files (.index per class)
│
├── 📄 requirements.txt
├── 📄 .gitignore
└── 📄 README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Face Detection + Alignment** | InsightFace Buffalo_SC (ONNX, CPUExecutionProvider) |
| **Face Recognition / Embedding** | InsightFace Buffalo_L — 512-D float32 embeddings |
| **Vector Search** | FAISS (per-school, per-class cosine similarity) |
| **Backend Framework** | FastAPI + Uvicorn (async, ASGI) |
| **Authentication** | JWT (python-jose) + bcrypt password hashing |
| **Database** | MySQL via mysql-connector-python (connection pool, size=5) |
| **CV Processing** | OpenCV, Pillow |
| **ML Runtime** | ONNX Runtime (CPU), TensorFlow, PyTorch CPU |
| **Frontend** | HTML5, CSS3, JavaScript |

---

## 🗄️ Database Schema

```
schools
  └── classes        (e.g. CS-A, MECH-B, per school)
        └── students       (name, roll_no, 3× 512-D embedding BLOBs)
        └── attendance     (one row per student per day + confidence_score)
        └── sessions       (groups all marks from one live camera session)

users                (admins + teachers, per school)
teacher_classes      (maps teachers → their assigned classes)
```

> Embeddings are stored as `MEDIUMBLOB` (3 angles per student) in MySQL **and** live in FAISS — the index can be fully rebuilt from the DB at any time.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- MySQL running locally (port 3306)
- Webcam or IP camera

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/Abhinav-gupta-123/AI-powered-attendance-management-system-.git
cd AI-powered-attendance-management-system-

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### First-Time Database Setup (run in order)

```bash
# Step 1 — Create all tables (schools, classes, students, attendance, sessions)
python scripts/DB_setup.py

# Step 2 — Migrate existing CSV data into MySQL (if you have prior data)
python scripts/migrate_csv.py

# Step 3 — Add users + teacher_classes tables and seed default admin account
python scripts/migrate_auth.py
# Default login → username: admin  |  password: admin123
# ⚠️  Change this immediately in production!

# Step 4 — Restructure FAISS indexes into per-school subdirectories
python scripts/migrate_faiss.py
```

### Run the Server

```bash
uvicorn scripts.api:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.
Interactive API docs at **http://localhost:8000/docs** (Swagger UI, auto-generated by FastAPI).

---

## 🔐 Auth & Roles

| Role | Access |
|---|---|
| **Admin** | Full access — all classes, all students, manage users |
| **Teacher** | Only their assigned classes (enforced at route level) |

JWT tokens expire after **12 hours**. Revoked on logout via in-memory blocklist.

---

## 🗺️ Roadmap

```
[✅] InsightFace Buffalo_SC/L detection + embedding pipeline
[✅] Per-school FAISS cosine similarity search
[✅] FastAPI + Uvicorn async backend
[✅] MySQL attendance logging with connection pooling
[✅] JWT auth — admin + teacher roles with class-level guards
[✅] Session-based attendance tracking with confidence scores
[✅] Student enrollment (3-angle embeddings in DB + FAISS)
[✅] Multi-school FAISS index structure
[✅] FAISS index inspector
[🔄] Anti-spoofing / liveness detection
[🔄] Analytics dashboard with charts
[🔄] Report export (CSV / PDF)
[🧪] Docker containerization
[🧪] Cloud deployment (AWS / GCP / Render)
[🧪] Email/SMS alerts for absent students
```

---

## 🤝 Contributing

This project is **actively under development** — new features and improvements ship regularly.

```bash
git checkout -b feature/your-feature-name
git commit -m "✨ Add: description"
git push origin feature/your-feature-name
# Open a Pull Request 🚀
```

---

## ⚠️ Work In Progress

- Anti-spoofing is actively being developed
- Currently CPU-only (GPU support is architecturally ready via ONNX providers)
- UI redesign is planned
- Cloud deployment coming soon

> This is a **living system** — it gets better every week. 🔥

---

## 📬 Contact

<div align="center">

**Abhinav Gupta**

[![GitHub](https://img.shields.io/badge/GitHub-Abhinav--gupta--123-181717?style=for-the-badge&logo=github)](https://github.com/Abhinav-gupta-123)

*Open to collaborations, feedback, and feature suggestions!*

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer&animation=fadeIn" width="100%"/>

### ⭐ If this project impressed you, drop a star — it means the world!

*Built with 💜 by Abhinav Gupta | Always improving, never stopping.*

</div>
