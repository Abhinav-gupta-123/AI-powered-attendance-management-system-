"""
FAISS Index Inspector
━━━━━━━━━━━━━━━━━━━━
Give it a class name → it reads the FAISS index + MySQL
and opens a browser visualization showing every enrolled
student with their details and vector count.

Usage:
  python scripts/inspect_index.py
  python scripts/inspect_index.py btech_cs_1
  python scripts/inspect_index.py CS-A

What it shows:
  - Summary cards  (students, vectors, class, index file size)
  - Per-student cards with name, roll, id, embedding count, norm values
  - Bar chart of embedding norms per student (visual quality indicator)
  - Vector similarity heatmap between all students in the class
"""

import os
import sys
import json
import webbrowser
import tempfile
import numpy as np

try:
    import mysql.connector
except ImportError:
    print("[ERROR] pip install mysql-connector-python")
    sys.exit(1)

try:
    import faiss
except ImportError:
    print("[ERROR] pip install faiss-cpu")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — must match Enroll_student.py
# ─────────────────────────────────────────────────────────────────────────────
from config import db_password
DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": f"{db_password}",   # <- change this
    "database": "attendance_system",
}
SCHOOL_ID  = 1
FAISS_DIR  = "faiss_indexes"


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def blob_to_emb(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def l2_norm(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb)
    return emb / n if n > 0 else emb


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_norm(a), l2_norm(b)))


def initials(name: str) -> str:
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper()


def avatar_color(student_id: int) -> str:
    colors = [
        "#185FA5", "#0F6E56", "#854F0B",
        "#993C1D", "#534AB7", "#993556",
    ]
    return colors[student_id % len(colors)]


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_class_data(class_name: str):
    index_path = os.path.join(FAISS_DIR, f"{class_name}.index")
    if not os.path.exists(index_path):
        print(f"[ERROR] Index not found: {index_path}")
        print(f"  Available indexes:")
        for f in os.listdir(FAISS_DIR):
            if f.endswith(".index"):
                print(f"    {f.replace('.index', '')}")
        sys.exit(1)

    print(f"[FAISS] Loading {index_path} ...")
    index     = faiss.read_index(index_path)
    n_vectors = index.ntotal
    file_size = os.path.getsize(index_path)
    print(f"[FAISS] {n_vectors} vectors  ({file_size/1024:.1f} KB)")

    print(f"[DB] Connecting to MySQL ...")
    conn = mysql.connector.connect(**DB_CONFIG)
    cur  = conn.cursor(dictionary=True)

    cur.execute(
        """SELECT s.id, s.name, s.roll_no, s.photo_path,
                  s.school_id, s.class_id, s.enrolled_at,
                  s.emb_1, s.emb_2, s.emb_3,
                  c.name AS class_name,
                  sc.name AS school_name
           FROM   students s
           JOIN   classes  c  ON c.id  = s.class_id
           JOIN   schools  sc ON sc.id = s.school_id
           WHERE  s.school_id = %s AND c.name = %s
           ORDER  BY s.id""",
        (SCHOOL_ID, class_name),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    print(f"[DB] {len(rows)} student(s) found in class '{class_name}'")

    students = []
    for row in rows:
        embs = []
        norms = []
        for key in ("emb_1", "emb_2", "emb_3"):
            if row[key]:
                emb = blob_to_emb(row[key])
                embs.append(emb)
                norms.append(round(float(np.linalg.norm(emb)), 2))

        students.append({
            "id":           int(row["id"]),
            "name":         str(row["name"]),
            "roll_no":      str(row["roll_no"]),
            "class_id":     int(row["class_id"]),
            "class_name":   str(row["class_name"]),
            "school_id":    int(row["school_id"]),
            "school_name":  str(row["school_name"]),
            "photo_path":   str(row["photo_path"] or "—"),
            "enrolled_at":  str(row["enrolled_at"]),
            "n_embeddings": len(embs),
            "norms":        norms,
            "avg_norm":     round(float(np.mean(norms)), 2) if norms else 0.0,
            "embeddings":   embs,
            "initials":     initials(str(row["name"])),
            "color":        avatar_color(int(row["id"])),
        })

    # Similarity matrix between students (using avg embedding)
    sim_matrix = []
    avg_embs   = []
    for s in students:
        if s["embeddings"]:
            avg_e = np.mean(np.stack(s["embeddings"]), axis=0)
            avg_embs.append(avg_e)
        else:
            avg_embs.append(np.zeros(512, dtype=np.float32))

    for i, a in enumerate(avg_embs):
        row_sims = []
        for j, b in enumerate(avg_embs):
            if i == j:
                row_sims.append(1.0)
            else:
                row_sims.append(round(cosine_sim(a, b), 4))
        sim_matrix.append(row_sims)

    return {
        "class_name":  class_name,
        "index_path":  index_path,
        "n_vectors":   n_vectors,
        "file_size_kb": round(file_size / 1024, 1),
        "students":    students,
        "sim_matrix":  sim_matrix,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  HTML REPORT
# ─────────────────────────────────────────────────────────────────────────────

def build_html(data: dict) -> str:
    students   = data["students"]
    sim_matrix = data["sim_matrix"]
    names      = [s["name"] for s in students]

    # Serialize for JS
    students_json  = json.dumps(students,  default=str)
    sim_matrix_json = json.dumps(sim_matrix)
    names_json     = json.dumps(names)

    # Build student cards HTML
    cards_html = ""
    for s in students:
        norm_bars = ""
        angle_labels = ["Straight", "Left", "Right"]
        for idx, norm in enumerate(s["norms"]):
            pct   = min(int((norm / 30.0) * 100), 100)
            label = angle_labels[idx] if idx < 3 else f"Angle {idx+1}"
            norm_bars += f"""
            <div style="margin-bottom:6px">
              <div style="display:flex;justify-content:space-between;
                          font-size:11px;color:#888;margin-bottom:2px">
                <span>{label}</span><span>norm {norm}</span>
              </div>
              <div style="background:#f0f0f0;border-radius:4px;height:6px">
                <div style="width:{pct}%;background:{s['color']};
                            border-radius:4px;height:6px"></div>
              </div>
            </div>"""

        cards_html += f"""
        <div style="background:#fff;border:0.5px solid #e0e0e0;
                    border-radius:12px;padding:16px;min-width:200px">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
            <div style="width:44px;height:44px;border-radius:50%;
                        background:{s['color']};display:flex;align-items:center;
                        justify-content:center;color:#fff;font-weight:500;
                        font-size:15px;flex-shrink:0">{s['initials']}</div>
            <div>
              <div style="font-weight:500;font-size:14px">{s['name']}</div>
              <div style="font-size:12px;color:#888">{s['roll_no']}</div>
            </div>
          </div>

          <table style="width:100%;border-collapse:collapse;
                        font-size:12px;margin-bottom:12px">
            <tr>
              <td style="color:#aaa;padding:3px 0;width:40%">MySQL id</td>
              <td style="color:#555;font-weight:500">{s['id']}</td>
            </tr>
            <tr>
              <td style="color:#aaa;padding:3px 0">FAISS id</td>
              <td style="color:#185FA5;font-weight:500">{s['id']} (same)</td>
            </tr>
            <tr>
              <td style="color:#aaa;padding:3px 0">School id</td>
              <td style="color:#555">{s['school_id']}</td>
            </tr>
            <tr>
              <td style="color:#aaa;padding:3px 0">School</td>
              <td style="color:#555">{s['school_name']}</td>
            </tr>
            <tr>
              <td style="color:#aaa;padding:3px 0">Class id</td>
              <td style="color:#555">{s['class_id']}</td>
            </tr>
            <tr>
              <td style="color:#aaa;padding:3px 0">Class</td>
              <td style="color:#555">{s['class_name']}</td>
            </tr>
            <tr>
              <td style="color:#aaa;padding:3px 0">Roll no</td>
              <td style="color:#555">{s['roll_no']}</td>
            </tr>
            <tr>
              <td style="color:#aaa;padding:3px 0">Photo path</td>
              <td style="color:#555;word-break:break-all">{s['photo_path']}</td>
            </tr>
            <tr>
              <td style="color:#aaa;padding:3px 0">Enrolled</td>
              <td style="color:#555">{str(s['enrolled_at'])[:19]}</td>
            </tr>
            <tr>
              <td style="color:#aaa;padding:3px 0">Vectors</td>
              <td style="color:#555">{s['n_embeddings']} stored</td>
            </tr>
          </table>

          <div style="font-size:11px;color:#888;margin-bottom:6px;font-weight:500;
                      border-top:0.5px solid #f0f0f0;padding-top:10px">
            Embedding norms  (healthy: 20–30)
          </div>
          {norm_bars}
          <div style="margin-top:8px;font-size:11px;color:#888">
            Avg norm: <strong style="color:#555">{s['avg_norm']}</strong>
          </div>
        </div>"""

    # Build similarity heatmap cells
    heatmap_headers = "".join(
        f'<th style="font-size:11px;padding:4px 8px;'
        f'font-weight:500;color:#888">{n.split()[0]}</th>'
        for n in names
    )
    heatmap_rows = ""
    for i, row_sims in enumerate(sim_matrix):
        cells = ""
        for j, sim in enumerate(row_sims):
            if i == j:
                bg   = "#e8f0fe"
                col  = "#185FA5"
                bold = "font-weight:500;"
            elif sim >= 0.6:
                bg   = "#fdecea"
                col  = "#a32d2d"
                bold = "font-weight:500;"
            elif sim >= 0.4:
                bg   = "#fff8e1"
                col  = "#854F0B"
                bold = ""
            else:
                bg   = "#f8f8f8"
                col  = "#888"
                bold = ""
            cells += (
                f'<td style="text-align:center;padding:6px 8px;'
                f'background:{bg};color:{col};font-size:12px;{bold}'
                f'border:0.5px solid #eee">{sim:.3f}</td>'
            )
        heatmap_rows += (
            f'<tr><td style="font-size:12px;padding:6px 10px;'
            f'font-weight:500;white-space:nowrap;color:#555;'
            f'border:0.5px solid #eee">'
            f'{names[i].split()[0]}</td>{cells}</tr>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FAISS Inspector — {data['class_name']}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0 }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f5f5f5; color: #222; padding: 24px }}
  h1   {{ font-size: 20px; font-weight: 500; margin-bottom: 4px }}
  h2   {{ font-size: 15px; font-weight: 500; margin: 24px 0 12px }}
  .meta {{ font-size: 13px; color: #888; margin-bottom: 24px }}
  .metrics {{ display: grid;
              grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
              gap: 12px; margin-bottom: 24px }}
  .metric  {{ background: #fff; border: 0.5px solid #e0e0e0;
              border-radius: 8px; padding: 14px }}
  .metric-label {{ font-size: 12px; color: #888; margin-bottom: 4px }}
  .metric-value {{ font-size: 22px; font-weight: 500 }}
  .cards {{ display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 16px; margin-bottom: 32px }}
  .section {{ background: #fff; border: 0.5px solid #e0e0e0;
              border-radius: 12px; padding: 20px; margin-bottom: 24px }}
  .legend {{ display: flex; gap: 16px; font-size: 12px; color: #888;
             margin-bottom: 12px; flex-wrap: wrap }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 2px;
                 display: inline-block; margin-right: 4px }}
  table {{ border-collapse: collapse; width: 100% }}
  th    {{ background: #f8f8f8; padding: 8px 10px;
           text-align: left; border: 0.5px solid #eee }}
</style>
</head>
<body>

<h1>FAISS index inspector — {data['class_name']}</h1>
<div class="meta">
  Index: {data['index_path']}
  &nbsp;&nbsp;|&nbsp;&nbsp;
  Size: {data['file_size_kb']} KB
  &nbsp;&nbsp;|&nbsp;&nbsp;
  {data['n_vectors']} total vectors in FAISS
</div>

<div class="metrics">
  <div class="metric">
    <div class="metric-label">Students enrolled</div>
    <div class="metric-value">{len(students)}</div>
  </div>
  <div class="metric">
    <div class="metric-label">Total vectors</div>
    <div class="metric-value">{data['n_vectors']}</div>
  </div>
  <div class="metric">
    <div class="metric-label">Vectors per student</div>
    <div class="metric-value">{round(data['n_vectors'] / max(len(students), 1), 1)}</div>
  </div>
  <div class="metric">
    <div class="metric-label">Index file size</div>
    <div class="metric-value">{data['file_size_kb']} KB</div>
  </div>
</div>

<h2>Enrolled students</h2>
<div class="cards">
  {cards_html}
</div>

<div class="section">
  <h2 style="margin-top:0">Embedding norm quality — all students</h2>
  <div style="font-size:12px;color:#888;margin-bottom:16px">
    Norm per stored angle. Healthy range: 20–30. Below 20 = possible occlusion during enrollment.
  </div>
  <div style="position:relative;height:{max(200, len(students) * 60)}px">
    <canvas id="normChart"></canvas>
  </div>
</div>

<div class="section">
  <h2 style="margin-top:0">Student similarity heatmap</h2>
  <div style="font-size:12px;color:#888;margin-bottom:12px">
    Cosine similarity between each pair of students (average embedding).
    High similarity between different students = potential confusion risk.
  </div>
  <div class="legend">
    <span><span class="legend-dot" style="background:#e8f0fe"></span>Self (1.000)</span>
    <span><span class="legend-dot" style="background:#fdecea"></span>High similarity ≥ 0.60 — review</span>
    <span><span class="legend-dot" style="background:#fff8e1"></span>Medium 0.40–0.59</span>
    <span><span class="legend-dot" style="background:#f8f8f8"></span>Low &lt; 0.40 — good separation</span>
  </div>
  <div style="overflow-x:auto">
    <table>
      <thead>
        <tr><th style="font-size:11px;padding:4px 8px"></th>{heatmap_headers}</tr>
      </thead>
      <tbody>{heatmap_rows}</tbody>
    </table>
  </div>
</div>

<script>
const students = {students_json};
const names    = {names_json};

const datasets = [];
const colors   = students.map(s => s.color);
const angleLabels = ['Straight', 'Left', 'Right'];

angleLabels.forEach((label, ai) => {{
  const data = students.map(s => s.norms[ai] || 0);
  datasets.push({{
    label,
    data,
    backgroundColor: colors.map(c => c + 'cc'),
    borderColor:     colors,
    borderWidth:     1,
    borderRadius:    4,
  }});
}});

new Chart(document.getElementById('normChart'), {{
  type: 'bar',
  data: {{ labels: names, datasets }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ position: 'top', labels: {{ font: {{ size: 12 }}, boxWidth: 10 }} }},
      tooltip: {{
        callbacks: {{
          label: ctx => ` norm ${{ctx.parsed.x.toFixed(1)}} (${{ctx.dataset.label}} angle)`
        }}
      }}
    }},
    scales: {{
      x: {{
        min: 0, max: 35,
        title: {{ display: true, text: 'Embedding norm', font: {{ size: 12 }} }},
        grid: {{ color: '#f0f0f0' }},
        ticks: {{ font: {{ size: 11 }} }},
      }},
      y: {{
        ticks: {{ font: {{ size: 12 }} }},
        grid: {{ display: false }},
      }}
    }}
  }}
}});
</script>

</body>
</html>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def list_available():
    if not os.path.exists(FAISS_DIR):
        print(f"[ERROR] faiss_indexes/ folder not found.")
        sys.exit(1)
    indexes = [f.replace(".index", "")
               for f in os.listdir(FAISS_DIR)
               if f.endswith(".index")]
    if not indexes:
        print("[ERROR] No .index files found in faiss_indexes/")
        sys.exit(1)
    return indexes


def main():
    available = list_available()

    if len(sys.argv) > 1:
        class_name = sys.argv[1]
    elif len(available) == 1:
        class_name = available[0]
        print(f"[INFO] Auto-selected only available class: {class_name}")
    else:
        print("\nAvailable classes:")
        for i, name in enumerate(available):
            print(f"  [{i+1}] {name}")
        choice = input("\nEnter class name or number: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                class_name = available[idx]
            else:
                print("[ERROR] Invalid number.")
                sys.exit(1)
        else:
            class_name = choice

    data = load_class_data(class_name)
    html = build_html(data)

    # Write to temp file and open in browser
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".html",
        delete=False, encoding="utf-8"
    )
    tmp.write(html)
    tmp.close()

    print(f"\n[OPEN] Opening browser: {tmp.name}")
    webbrowser.open(f"file://{tmp.name}")

    print("\n" + "=" * 56)
    print(f"  Class        : {data['class_name']}")
    print(f"  Students     : {len(data['students'])}")
    print(f"  FAISS vectors: {data['n_vectors']}")
    print(f"  Index size   : {data['file_size_kb']} KB")
    print("=" * 56)
    for s in data["students"]:
        print(f"\n  {'─'*52}")
        print(f"  id          : {s['id']}  (MySQL = FAISS, permanent)")
        print(f"  name        : {s['name']}")
        print(f"  roll_no     : {s['roll_no']}")
        print(f"  school      : {s['school_name']} (id={s['school_id']})")
        print(f"  class       : {s['class_name']} (id={s['class_id']})")
        print(f"  photo_path  : {s['photo_path']}")
        print(f"  enrolled_at : {s['enrolled_at']}")
        print(f"  vectors     : {s['n_embeddings']}  avg_norm={s['avg_norm']}")
        print(f"  norms       : {s['norms']}")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()