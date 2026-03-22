<div align="center">

<!-- ANIMATED HEADER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=FaceAttend%20AI&fontSize=60&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Next-Gen%20Attendance%20%7C%20Powered%20by%20Computer%20Vision&descAlignY=60&descSize=18" width="100%"/>

<!-- BADGES -->
<p align="center">
  <img src="https://img.shields.io/badge/Status-🔥%20Actively%20Developing-brightgreen?style=for-the-badge&logo=github"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-Face%20Detection-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-FF6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/MySQL-Database-4479A1?style=for-the-badge&logo=mysql&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLO-Detection-00FFFF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/DeepFace-Recognition-FF4500?style=for-the-badge"/>
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

**FaceAttend AI** is a cutting-edge, production-grade attendance management system that uses **computer vision + deep learning** to identify people from a live camera feed and automatically record their attendance — in real time, with high accuracy.

No more manual entries. No more buddy punching. No more spreadsheets.

Just walk in front of the camera → recognized → ✅ **Done.**

---

## 🧠 The Tech Brain

```
📸 Camera Feed
      ↓
🎯 YOLOv8 (Face Detection)  ←── Ultralytics
      ↓
🧬 AdaFace / DeepFace / InsightFace (Face Embedding)
      ↓
⚡ FAISS Vector Index (Lightning-fast Similarity Search)
      ↓
🆔 Identity Matched
      ↓
🗄️  MySQL Database (Attendance Logged)
      ↓
🌐 Flask API → HTML Frontend Dashboard
```

---

## 🔥 Features

| Feature | Status |
|---|---|
| 🎯 Real-time face detection via YOLOv8 | ✅ Live |
| 🧬 Deep face embedding (AdaFace/InsightFace/DeepFace) | ✅ Live |
| ⚡ FAISS-powered sub-millisecond face search | ✅ Live |
| 🏫 Multi-school / multi-organization support | ✅ Live |
| 🗄️ MySQL attendance records & logs | ✅ Live |
| 🌐 Web dashboard (Flask + HTML) | ✅ Live |
| 📦 Register new students/employees via webcam | ✅ Live |
| 🚫 Anti-spoofing / liveness detection | 🔄 In Progress |
| 📊 Analytics & attendance reports | 🔄 In Progress |
| 📱 Mobile-friendly UI | 🔄 In Progress |
| 🔔 Email/SMS alert on absence | 🧪 Planned |
| ☁️ Cloud deployment (Docker + CI/CD) | 🧪 Planned |

---

## 🏗️ Project Architecture

```
AI-powered-attendance-management-system/
│
├── 📁 frontend/              # HTML/CSS/JS Web Dashboard
│   └── ...                  # Attendance UI, Registration forms
│
├── 📁 models/                # Pre-trained AI model weights
│   └── ...                  # AdaFace, YOLO face models
│
├── 📁 scripts/               # Core logic & utilities
│   └── ...                  # Detection, embedding, registration
│
├── 📁 faiss_indexes/         # Per-school FAISS vector databases
│   └── Default School/      # Stores face embeddings per org
│
├── 📄 requirements.txt       # All dependencies
├── 📄 .gitignore
└── 📄 README.md              # You are here 😎
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| **Face Detection** | YOLOv8 (Ultralytics), MTCNN, RetinaFace |
| **Face Recognition** | DeepFace, AdaFace, InsightFace |
| **Vector Search** | FAISS (Facebook AI Similarity Search) |
| **Backend** | Python, Flask, Flask-CORS, Gunicorn |
| **Database** | MySQL (mysql-connector-python) |
| **CV Processing** | OpenCV, Pillow, scikit-image |
| **ML Frameworks** | TensorFlow, PyTorch (CPU), Keras, ONNX Runtime |
| **Utilities** | NumPy, Pandas, Polars, SciPy, scikit-learn |
| **Frontend** | HTML5, CSS3, JavaScript |

</div>

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- MySQL installed and running
- Webcam / IP Camera

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/Abhinav-gupta-123/AI-powered-attendance-management-system-.git
cd AI-powered-attendance-management-system-

# 2. Create a virtual environment (highly recommended!)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Set up your MySQL database
# Create a database and update the config in your script
# (DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)

# 5. Run the Flask server
python scripts/app.py        # or whatever your entry point is

# 6. Open the dashboard
# Navigate to http://localhost:5000 in your browser
```

---

## 🎬 How It Works

### 1️⃣ Register a Face
> Open the dashboard → Add student/employee → Capture photos via webcam → System generates a **512-D face embedding** using AdaFace/InsightFace → Stored in the **FAISS vector index** for that school/org.

### 2️⃣ Mark Attendance (Real-Time)
> Camera feed starts → **YOLOv8** detects faces in each frame → Each detected face is embedded → **FAISS** performs nearest-neighbor search → If similarity > threshold → Person identified → **MySQL** logs attendance with timestamp.

### 3️⃣ View Records
> Web dashboard shows live attendance, history, and per-student records.

---

## 📸 Architecture Diagrams

> The repo includes detailed SVG architecture diagrams:
> - `final_adaface_architecture_svg.svg` — AdaFace model pipeline
> - `final_attendance_architecture.svg` — Full system flow

*(Check them out in the repo root!)*

---

## 🧪 Why These Models?

| Model | Why It's Here |
|---|---|
| **YOLOv8** | Real-time, blazing-fast face detection even on CPU |
| **AdaFace** | State-of-the-art face recognition, especially for low-quality/partial faces |
| **InsightFace** | Production-proven face analysis & embedding |
| **DeepFace** | Wrapper for multiple recognition models — flexible |
| **FAISS** | Can search millions of face embeddings in milliseconds |
| **MTCNN / RetinaFace** | Accurate face alignment for better embedding quality |

---

## 🗺️ Roadmap

```
[✅] Core face detection + recognition pipeline
[✅] FAISS-based fast identity lookup
[✅] Flask REST API
[✅] MySQL attendance logging
[✅] Multi-school FAISS indexes
[✅] Web dashboard
[🔄] Anti-spoofing / liveness detection
[🔄] Analytics dashboard with charts
[🔄] Report export (CSV / PDF)
[🧪] Docker containerization
[🧪] Cloud deployment (AWS / GCP / Render)
[🧪] Email/SMS alerts for absent students
[🧪] Mobile app or PWA
```

---

## 🤝 Contributing

This project is **actively under development** — improvements and new features are being added regularly!

If you want to contribute:

```bash
# Fork the repo, create a branch
git checkout -b feature/your-feature-name

# Make your changes, then
git commit -m "✨ Add: your feature description"
git push origin feature/your-feature-name

# Open a Pull Request 🚀
```

---

## ⚠️ Known Limitations / Work In Progress

- Anti-spoofing (photo/video attacks) is being actively worked on
- GPU support is present in architecture but CPU-mode is the default
- UI is functional but a redesign is planned
- Cloud deployment is upcoming

> This project is a **living, breathing system** — things get better every week! 🔥

---

## 📬 Contact

<div align="center">

**Abhinav Gupta**

[![GitHub](https://img.shields.io/badge/GitHub-Abhinav--gupta--123-181717?style=for-the-badge&logo=github)](https://github.com/Abhinav-gupta-123)

*Open to collaborations, feedback, and feature suggestions!*

</div>

---

## 📄 License

This project is open-source. Feel free to fork, star ⭐, and build upon it!

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer&animation=fadeIn" width="100%"/>

### ⭐ If this project helped you or impressed you, drop a star — it means the world!

*Built with 💜 by Abhinav Gupta | Always improving, never stopping.*

</div>
