# app.py
from flask import Flask, render_template_string, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pickle, os
from PIL import Image
import numpy as np
import face_recognition

# 🔹 YOLO import + model yükleme
from ultralytics import YOLO
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL", "best.pt")  # repo köküne koyduğun best.pt
yolo = YOLO(YOLO_MODEL_PATH)

# --- Flask app ---
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "yoklama123")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB payload limiti

# --- DB URL (Heroku + local fallback) ---
db_url = os.environ.get("DATABASE_URL")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
if not db_url:
    db_url = "sqlite:///app.db"  # add-on yoksa SQLite kullan (ephemeral)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Heroku Postgres çoğunlukla SSL ister
if db_url.startswith("postgresql://"):
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "connect_args": {"sslmode": "require"},
        "pool_pre_ping": True,
    }

# --- DB ---
db = SQLAlchemy(app)

# Model
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.String(32))
    name = db.Column(db.String(100))
    entry_time = db.Column(db.DateTime)
    exit_time = db.Column(db.DateTime)
    duration = db.Column(db.Interval)

# ------------- CONFIDENCE HESAPLAMA -------------
def face_confidence(face_distance, match_threshold=0.45):
    """
    face_distance -> face_recognition.face_distance() çıktısı
    match_threshold -> compare_faces(tolerance) ile uyumlu eşik
    Çıktı: yüzdelik güven skoru (0-100)
    """
    range_val = (1.0 - match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > match_threshold:
        return round(max(0.0, min(1.0, linear_val)) * 100, 2)
    else:
        # Yakın mesafelerde daha keskin yükselen bir eğri
        value = (linear_val + ((1.0 - linear_val) * pow((linear_val - 0.5) * 2, 0.2)))
        return round(max(0.0, min(1.0, value)) * 100, 2)

# 🔹 YOLO ile yüz tespiti (xyxy + det_conf%)
def detect_faces_yolo(img_array, conf_thr=0.4, imgsz=640):
    res = yolo.predict(source=img_array, conf=conf_thr, imgsz=imgsz, verbose=False)
    boxes = []
    for r in res:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            det_conf = float(b.conf[0].item()) * 100.0  # %
            boxes.append((x1, y1, x2, y2, det_conf))
    return boxes

# Sağlık kontrolü
@app.route("/health")
def health():
    return "OK", 200

# İlk tablo kurulumu – sadece elle çağır
@app.route("/initdb")
def initdb():
    with app.app_context():
        db.create_all()
    return "DB OK", 200

# ----------------- ORTAK STİL (tek mavi tema) -----------------
BASE_CSS = """
<style>
  :root{
    --bg1:#eaf4ff; --bg2:#f6fbff; --text:#0b2a55; --nav:#ffffffcc;
    --btnMain:#0ea5e9; --btnMainH:#0284c7; --btnOutline:#cfe6ff;
    --in:#28a745; --inH:#218838; --out:#dc3545; --outH:#c82333;
  }
  body{ background:linear-gradient(180deg,var(--bg1),var(--bg2)); color:var(--text); }
  .page-wrap{ min-height:100vh; display:flex; flex-direction:column; }
  .navbar{ background:var(--nav); backdrop-filter:blur(8px); }
  .brand{ font-weight:700; color:var(--text)!important; }
  .top-actions .btn{ padding:.9rem 1.25rem; font-weight:700; border-radius:12px; }
  .btn-primary{ background:var(--btnMain); border:0; }
  .btn-primary:hover{ background:var(--btnMainH); }
  .btn-outline{ color:var(--text); border:1px solid var(--btnOutline); background:#fff; }
  .btn-outline:hover{ background:#f0f7ff; }
  .hero{ flex:1; display:flex; align-items:center; justify-content:center; flex-wrap:wrap; gap:2rem; }
  .big-square{
    width:clamp(180px,18vw,240px); height:clamp(180px,18vw,240px);
    border-radius:22px; display:grid; place-items:center; font-weight:800;
    box-shadow:0 10px 24px rgba(0,0,0,.25);
  }
  .sq-in{ background:var(--in); color:#fff; } .sq-in:hover{ background:var(--inH); }
  .sq-out{ background:var(--out); color:#fff; } .sq-out:hover{ background:var(--outH); }
  #cameraArea{ display:none; margin-top:30px; }
  .cam-row{ gap:1.25rem; }
  .cam-card{
    background:#fff; border:1px solid #e6eefc; color:var(--text);
    border-radius:16px; padding:16px;
  }
  #video, #preview{
    border-radius:12px; width:100%; height:auto;
    aspect-ratio:16/9; object-fit:cover; max-height:420px;
  }
  #flashEffect{
    display:none; position:fixed; inset:0; background:#fff; z-index:9999; opacity:1;
    animation: flash-pop .25s ease;
  }
  @keyframes flash-pop{
    0%{opacity:1} 100%{opacity:0}
  }
</style>
"""

# ----------------- HOME -----------------
HOME_HTML = f'''
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <title>Attendance</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  {BASE_CSS}
</head>
<body>
  <div id="flashEffect"></div>
  <div class="page-wrap">
    <nav class="navbar navbar-expand px-3">
      <div class="container-fluid">
        <a class="navbar-brand brand" href="/">Attendance</a>
        <div class="top-actions d-flex gap-2 ms-auto">
          <a href="{{{{ url_for('dashboard') }}}}" class="btn btn-primary">Dashboard</a>
          <a href="{{{{ url_for('add_user') }}}}" class="btn btn-outline">Add User</a>
        </div>
      </div>
    </nav>

    <main class="container py-4">
      <div class="hero">
        <button class="big-square sq-in border-0" onclick="startCamera('entrance')">
          <div class="text-center">
            <div style="font-size:42px; line-height:1;">📥</div>
            <div style="font-size:20px;">GİRİŞ</div>
          </div>
        </button>
        <button class="big-square sq-out border-0" onclick="startCamera('exit')">
          <div class="text-center">
            <div style="font-size:42px; line-height:1;">📤</div>
            <div style="font-size:20px;">ÇIKIŞ</div>
          </div>
        </button>
      </div>

      <div id="cameraArea" class="container" style="display:none; margin-top:30px;">
        <div class="row cam-row justify-content-center align-items-start">
          <div class="col-12 col-lg-5">
            <div class="cam-card">
              <video id="video" autoplay playsinline></video>
              <div class="mt-3 d-flex gap-2 justify-content-center">
                <button id="snap" class="btn btn-primary">📸 Fotoğraf Çek & Kaydet</button>
                <form id="photoForm" class="d-inline">
                  <input type="hidden" name="action" id="currentAction" value="/attendance_photo">
                </form>
              </div>
            </div>
          </div>
          <div class="col-12 col-lg-5">
            <div class="cam-card">
              <img id="preview" src="" style="display:none;">
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>

  <script>
    function startCamera(type) {{
      document.getElementById('cameraArea').style.display = 'block';
      const act = (type === 'entrance') ? '/attendance_photo' : '/exit_photo';
      document.getElementById('currentAction').value = act;

      navigator.mediaDevices.getUserMedia({{
        video: {{ width: {{ ideal: 640 }}, height: {{ ideal: 360 }}, facingMode: "user" }}
      }})
      .then(stream => {{
        const v = document.getElementById('video');
        v.srcObject = stream;
      }})
      .catch(err => {{
        alert('Kamera erişimi reddedildi: ' + err);
      }});

      document.getElementById('cameraArea').scrollIntoView({{behavior:'smooth', block:'center'}});
    }}

    document.addEventListener('DOMContentLoaded', () => {{
      const snap = document.getElementById('snap');
      if (!snap) return;

      snap.onclick = async function(e) {{
        e.preventDefault();
        const video = document.getElementById('video');
        const W = 640;
        const vw = video.videoWidth || 960;
        const vh = video.videoHeight || 540;
        const H = Math.round(W * vh / vw);

        const canvas = document.createElement('canvas');
        canvas.width = W; canvas.height = H;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, W, H);

        const fx = document.getElementById('flashEffect');
        fx.style.display = 'block';
        setTimeout(() => fx.style.display = 'none', 200);

        canvas.toBlob(async (blob) => {{
          if (!blob) {{ alert("Görüntü yakalanamadı."); return; }}
          const prev = document.getElementById('preview');
          prev.src = URL.createObjectURL(blob); prev.style.display = 'block';

          const fd = new FormData(); fd.append('photo', blob, 'frame.jpg');
          const url = document.getElementById('currentAction').value;

          try {{
            const res = await fetch(url, {{ method: 'POST', body: fd }});
            const ct = res.headers.get('content-type') || '';
            if (!res.ok) {{
              const txt = await res.text(); alert("Gönderim hatası: " + res.status + " " + txt); return;
            }}
            if (ct.includes('application/json')) {{
              const data = await res.json();
              if (data && data.name) {{
                alert('📸 ' + (data.action || 'İşlem') + ' → ' + data.name);
              }} else {{
                alert('📸 Fotoğraf çekildi.');
              }}
            }} else {{
              alert('📸 Fotoğraf çekildi.');
            }}
            window.location.href = "/";
          }} catch (err) {{ alert("Ağ hatası: " + err); }}
        }}, 'image/jpeg', 0.7);
      }};
    }});
  </script>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

# ----------------- DASHBOARD -----------------
DASHBOARD_HTML = f'''
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <title>Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  {BASE_CSS}
</head>
<body>
  <div class="page-wrap">
    <nav class="navbar navbar-expand px-3">
      <div class="container-fluid">
        <a class="navbar-brand brand" href="/">Attendance</a>
        <div class="top-actions d-flex gap-2 ms-auto">
          <a href="{{{{ url_for('dashboard') }}}}" class="btn btn-primary">Dashboard</a>
          <a href="{{{{ url_for('add_user') }}}}" class="btn btn-outline">Add User</a>
        </div>
      </div>
    </nav>

    <main class="container py-4">
      <div class="card p-3" style="background:#fff; border:1px solid #e6eefc; border-radius:16px;">
        <h3 class="mb-3">Attendance Table</h3>
        <div class="table-responsive">
          <table class="table table-hover table-bordered align-middle">
            <thead>
              <tr><th>ID</th><th>Person ID</th><th>Name</th><th>Entry</th><th>Exit</th><th>Duration</th></tr>
            </thead>
            <tbody>
            {{% for r in data %}}
              <tr>
                <td>{{{{r.id}}}}</td>
                <td>{{{{r.person_id}}}}</td>
                <td>{{{{r.name}}}}</td>
                <td>{{{{r.entry_time or ''}}}}</td>
                <td>{{{{r.exit_time or ''}}}}</td>
                <td>{{{{r.duration or ''}}}}</td>
              </tr>
            {{% endfor %}}
            </tbody>
          </table>
        </div>
      </div>
    </main>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

# ----------------- ADD USER -----------------
ADD_USER_HTML = f'''
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <title>Add User</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  {BASE_CSS}
</head>
<body>
  <div class="page-wrap">
    <nav class="navbar navbar-expand px-3">
      <div class="container-fluid">
        <a class="navbar-brand brand" href="/">Attendance</a>
        <div class="top-actions d-flex gap-2 ms-auto">
          <a href="{{{{ url_for('dashboard') }}}}" class="btn btn-primary">Dashboard</a>
          <a href="{{{{ url_for('add_user') }}}}" class="btn btn-outline">Add User</a>
        </div>
      </div>
    </nav>

    <main class="container py-4">
      <div class="card p-4" style="background:#fff; border:1px solid #e6eefc; border-radius:16px;">
        <h3 class="mb-3">Add New User</h3>
        <form method="post" enctype="multipart/form-data">
          <div class="mb-3">
            <label class="form-label">Name</label>
            <input type="text" class="form-control" name="username" required>
          </div>
          <div class="mb-3">
            <label class="form-label">Face Image</label>
            <input type="file" class="form-control" name="face_image" accept="image/*" required>
          </div>
          <button class="btn btn-primary" type="submit">Add</button>
        </form>
      </div>
    </main>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

# ----------------- ROUTES -----------------
@app.route('/', methods=['GET'])
def home():
    return render_template_string(HOME_HTML)

@app.route('/dashboard')
def dashboard():
    data = Attendance.query.order_by(Attendance.entry_time.desc()).all()
    return render_template_string(DASHBOARD_HTML, data=data)

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        file = request.files.get('face_image')
        if not username:
            return redirect(url_for('add_user'))
        if not file:
            return redirect(url_for('add_user'))

        # Görseli yükle
        try:
            img = Image.open(file.stream).convert("RGB")
        except Exception:
            return redirect(url_for('add_user'))

        img_np = np.array(img)

        # 🔹 Yüzü YOLO ile bul (en güvenilir kutuyu seç)
        boxes = detect_faces_yolo(img_np, conf_thr=0.35, imgsz=640)
        if not boxes:
            return redirect(url_for('add_user'))
        # en yüksek det_conf olan kutu
        x1, y1, x2, y2, detc = max(boxes, key=lambda b: b[4])

        # face_recognition için (top, right, bottom, left) formatı
        fr_loc = [(y1, x2, y2, x1)]
        encs = face_recognition.face_encodings(img_np, known_face_locations=fr_loc)
        if not encs:
            return redirect(url_for('add_user'))
        enc = encs[0]

        # Basit pickle veritabanı (ephemeral)
        if os.path.exists("face_db.pickle"):
            with open("face_db.pickle", "rb") as f:
                encodings, names, ids = pickle.load(f)
        else:
            encodings, names, ids = [], [], []
        new_id = f"{len(ids)+1:03d}"
        encodings.append(enc)
        names.append(username)
        ids.append(new_id)
        with open("face_db.pickle", "wb") as f:
            pickle.dump((encodings, names, ids), f)

        return redirect(url_for('add_user'))

    return render_template_string(ADD_USER_HTML)

# ----------------- FOTOĞRAF İŞLEME -----------------
@app.route('/attendance_photo', methods=['POST'])
def attendance_photo():
    return process_photo(is_entry=True)

@app.route('/exit_photo', methods=['POST'])
def exit_photo():
    return process_photo(is_entry=False)

def process_photo(is_entry: bool):
    """
    JPEG Blob (multipart/form-data) bekler: field adı 'photo'
    JSON döner: {status, action, name, confidence, recognized, person_id?}
    """
    file = request.files.get('photo')
    if not file:
        return jsonify({"status": "error", "message": "No photo"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"status": "error", "message": "Invalid image"}), 400

    img_array = np.array(image)
    action_text = "Giriş" if is_entry else "Çıkış"

    # 🔹 YOLO ile yüz tespiti
    boxes = detect_faces_yolo(img_array, conf_thr=0.4, imgsz=640)
    if not boxes:
        return jsonify({
            "status": "ok",
            "action": action_text,
            "name": "Yüz bulunamadı",
            "confidence": 0.0,
            "recognized": False
        }), 200

    if not os.path.exists("face_db.pickle"):
        return jsonify({
            "status": "ok",
            "action": "Görüntü",
            "name": "Veritabanı boş",
            "confidence": 0.0,
            "recognized": False
        }), 200

    with open("face_db.pickle", "rb") as f:
        known_encodings, known_names, known_ids = pickle.load(f)

    tolerance = 0.45
    best = None

    for (x1, y1, x2, y2, det_conf) in boxes:
        # face_recognition için (top, right, bottom, left)
        fr_loc = [(y1, x2, y2, x1)]
        encs = face_recognition.face_encodings(img_array, known_face_locations=fr_loc)
        if not encs:
            continue

        distances = face_recognition.face_distance(known_encodings, encs[0])
        if len(distances) == 0:
            continue

        idx = int(np.argmin(distances))
        dist = float(distances[idx])
        rec_conf = face_confidence(dist, match_threshold=tolerance)  # % değer

        if (best is None) or (rec_conf > best["rec_conf"]):
            best = {"idx": idx, "dist": dist, "rec_conf": rec_conf, "det_conf": det_conf}

    if not best:
        return jsonify({
            "status": "ok",
            "action": action_text,
            "name": "Unknown (0%)",
            "confidence": 0.0,
            "recognized": False
        }), 200

    is_match = best["dist"] <= tolerance
    name_only = known_names[best["idx"]]

    if is_match:
        person_id = known_ids[best["idx"]]
        now = datetime.now()

        last_record = Attendance.query.filter_by(person_id=person_id).order_by(Attendance.entry_time.desc()).first()
        if last_record and (
            (is_entry and last_record.entry_time and (now - last_record.entry_time) < timedelta(hours=2)) or
            ((not is_entry) and last_record.exit_time and (now - last_record.exit_time) < timedelta(hours=2))
        ):
            return jsonify({
                "status": "ok",
                "action": action_text,
                "name": f"{name_only} ({best['rec_conf']}%) - det:{best['det_conf']:.1f}% - Tekrarlı işlem engellendi",
                "confidence": best["rec_conf"],
                "recognized": True,
                "person_id": person_id
            }), 200

        if is_entry:
            yeni = Attendance(person_id=person_id, name=name_only, entry_time=now)
            db.session.add(yeni); db.session.commit()
        else:
            if last_record and last_record.exit_time is None:
                last_record.exit_time = now
                last_record.duration = last_record.exit_time - last_record.entry_time
                db.session.commit()

        return jsonify({
            "status": "ok",
            "action": action_text,
            "name": f"{name_only} ({best['rec_conf']}%) - det:{best['det_conf']:.1f}%",
            "confidence": best["rec_conf"],
            "recognized": True,
            "person_id": person_id
        }), 200

    else:
        return jsonify({
            "status": "ok",
            "action": action_text,
            "name": f"Unknown ({best['rec_conf']}%) - det:{best['det_conf']:.1f}%",
            "confidence": best["rec_conf"],
            "recognized": False
        }), 200

# ----------------- MAIN -----------------
if __name__ == '__main__':
    # Lokal geliştirme için. Heroku'da Gunicorn Procfile ile başlatır.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
