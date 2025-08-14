from flask import Flask, render_template_string, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pickle, base64, os
from io import BytesIO
from PIL import Image
import numpy as np
import face_recognition

# --- YOLO: sadece yüz TESPİTİ için kullanıyoruz ---
from ultralytics import YOLO

# ================== AYARLAR ==================
# YOLO model dosya yolu (örn: "runs/detect/train/weights/best.pt")
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "/Users/apple/Documents/GitHub/Heroku-Attendance/best.pt")
YOLO_CONF = 0.45  # tespit eşiği

# Modelinde birden fazla sınıf varsa ve yüz sınıfının adını biliyorsan doldur:
# Örn: {"face", "person_face"}
YOLO_FACE_CLASS_NAMES = None  # Sadece yüz sınıflarını bırakmak için set ver (yoksa tüm box'ları yüz varsayar)

# face_recognition karşılaştırma eşiği (daha küçük = daha katı)
FR_TOLERANCE = 0.45
# =============================================

app = Flask(__name__)
app.secret_key = 'yoklama123'

# PostgreSQL örneği
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://emre:1234@localhost/attendance_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.String(32))
    name = db.Column(db.String(100))
    entry_time = db.Column(db.DateTime)
    exit_time = db.Column(db.DateTime)
    duration = db.Column(db.Interval)

with app.app_context():
    db.create_all()

# --- YOLO modeli yükle ---
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"[INFO] YOLO yüklendi: {YOLO_MODEL_PATH}")
except Exception as e:
    yolo_model = None
    print(f"[WARN] YOLO modeli yüklenemedi: {e}")

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
  .alert{ border-radius:12px; }
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
      {{% with messages = get_flashed_messages() %}}
        {{% if messages %}}
          <div class="alert alert-success alert-dismissible fade show" role="alert">
            {{{{ messages[0] }}}}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
          </div>
        {{% endif %}}
      {{% endwith %}}

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
              <video id="video" autoplay></video>
              <div class="mt-3 d-flex gap-2 justify-content-center">
                <button id="snap" class="btn btn-primary">📸 Fotoğraf Çek</button>
                <form id="photoForm" method="POST" enctype="multipart/form-data" class="d-inline">
                  <input type="hidden" name="imgData" id="imgData">
                  <button type="submit" class="btn btn-success" id="sendBtn" disabled>Kaydet</button>
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
      document.getElementById('sendBtn').disabled = true;
      const form = document.getElementById('photoForm');
      form.action = (type === 'entrance') ? '/attendance_photo' : '/exit_photo';
      navigator.mediaDevices.getUserMedia({{ video: true }}).then(stream => {{
        document.getElementById('video').srcObject = stream;
      }});
      document.getElementById('cameraArea').scrollIntoView({{behavior:'smooth', block:'center'}});
    }}

    document.addEventListener('DOMContentLoaded', () => {{
      const snap = document.getElementById('snap');
      if (snap) {{
        snap.onclick = function(e) {{
          e.preventDefault();
          var canvas = document.createElement('canvas');
          var video = document.getElementById('video');
          canvas.width = video.videoWidth || 960;
          canvas.height = video.videoHeight || 540;
          canvas.getContext('2d').drawImage(video, 0, 0);
          var dataUrl = canvas.toDataURL('image/png');
          document.getElementById('imgData').value = dataUrl;
          const prev = document.getElementById('preview');
          prev.src = dataUrl; prev.style.display = 'block';
          document.getElementById('sendBtn').disabled = false;
          alert("📸 Fotoğraf çekildi!");
        }};
      }}
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
      {{% with messages = get_flashed_messages() %}}
        {{% if messages %}}
          <div class="alert alert-success alert-dismissible fade show" role="alert">
            {{{{ messages[0] }}}}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
          </div>
        {{% endif %}}
      {{% endwith %}}

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
        username = request.form['username']
        file = request.files.get('face_image')
        if not file:
            flash("Fotoğraf yüklenmedi.")
            return redirect(url_for('add_user'))
        path = f"tmp_{username}.jpg"
        file.save(path)
        img = face_recognition.load_image_file(path)  # RGB np.array
        face_locs = face_recognition.face_locations(img)
        if not face_locs:
            os.remove(path)
            flash("Yüz algılanamadı.")
            return redirect(url_for('add_user'))
        enc = face_recognition.face_encodings(img, face_locs)[0]
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
        os.remove(path)
        flash(f"Kullanıcı eklendi: {username} (ID: {new_id})")
        return redirect(url_for('add_user'))
    return render_template_string(ADD_USER_HTML)

# --------- YOLO ile YÜZ TESPİTİ yardımcıları ---------
def detect_faces_yolo(img_rgb: np.ndarray):
    """
    Ultralytics YOLO'dan gelen kutuları face_recognition formatına çevirir.
    face_recognition: (top, right, bottom, left)
    YOLO: (x1, y1, x2, y2)
    """
    if yolo_model is None:
        return []

    # YOLO tahmini
    results = yolo_model.predict(img_rgb, conf=YOLO_CONF, verbose=False)
    if not results or len(results) == 0:
        return []

    boxes = []
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []

    # Sınıf filtresi gerekiyorsa uygula
    # (Ultralytics'te class isimleri res.names içinde tutulur)
    names_map = getattr(res, "names", None)  # dict: class_id -> class_name
    for i in range(len(res.boxes)):
        b = res.boxes[i]
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cls_ok = True
        if YOLO_FACE_CLASS_NAMES and names_map is not None:
            cls_id = int(b.cls[0].item()) if b.cls is not None else None
            cls_name = names_map.get(cls_id, None) if cls_id is not None else None
            cls_ok = cls_name in YOLO_FACE_CLASS_NAMES
        if not cls_ok:
            continue

        h, w = img_rgb.shape[:2]
        x1 = max(0, min(int(x1), w-1))
        x2 = max(0, min(int(x2), w-1))
        y1 = max(0, min(int(y1), h-1))
        y2 = max(0, min(int(y2), h-1))
        top, right, bottom, left = y1, x2, y2, x1
        boxes.append((top, right, bottom, left))

    return boxes

# ----------------- FOTOĞRAF İŞLEME (YOLO tespit + FR tanıma) -----------------
@app.route('/attendance_photo', methods=['POST'])
def attendance_photo():
    return process_photo(is_entry=True)

@app.route('/exit_photo', methods=['POST'])
def exit_photo():
    return process_photo(is_entry=False)

def process_photo(is_entry: bool):
    img_data = request.form.get('imgData')
    if not img_data:
        flash("Fotoğraf alınamadı.")
        return redirect(url_for('home'))

    img_bytes = base64.b64decode(img_data.split(',')[1])
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_rgb = np.array(image)  # face_recognition RGB ister

    # 1) YOLO ile yüz kutularını bul
    face_locs = detect_faces_yolo(img_rgb)

    if not face_locs:
        flash("Yüz algılanamadı (YOLO).")
        return redirect(url_for('home'))

    # 2) Kutulardan embedding çıkar
    face_encs = face_recognition.face_encodings(img_rgb, face_locs)
    if not face_encs:
        flash("Yüz embedding çıkarılamadı.")
        return redirect(url_for('home'))

    # 3) Veritabanı yükle
    if not os.path.exists("face_db.pickle"):
        flash("Yüz veritabanı yok.")
        return redirect(url_for('home'))

    with open("face_db.pickle", "rb") as f:
        known_encodings, known_names, known_ids = pickle.load(f)

    # 4) Her yüz için eşleşme dene (ilk bulunan eşleşme ile devam)
    matched = None
    for enc in face_encs:
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=FR_TOLERANCE)
        if True in matches:
            idx = matches.index(True)
            matched = (known_ids[idx], known_names[idx])
            break

    if not matched:
        flash("Yüz tanınamadı (eşleşme yok).")
        return redirect(url_for('home'))

    person_id, name = matched
    now = datetime.now()

    last_record = Attendance.query.filter_by(person_id=person_id).order_by(Attendance.entry_time.desc()).first()
    # 2 saat içinde tekrar aynı işlem engeli
    if last_record and (
        (is_entry and last_record.entry_time and (now - last_record.entry_time) < timedelta(hours=2)) or
        (not is_entry and last_record.exit_time and (now - last_record.exit_time) < timedelta(hours=2))
    ):
        flash(f"{name}, 2 saat içinde tekrar kayıt yapılamaz.")
        return redirect(url_for('home'))

    if is_entry:
        yeni = Attendance(person_id=person_id, name=name, entry_time=now)
        db.session.add(yeni)
        db.session.commit()
        flash(f"Hoş geldin {name}, giriş kaydedildi.")
    else:
        if last_record and last_record.exit_time is None:
            last_record.exit_time = now
            last_record.duration = last_record.exit_time - last_record.entry_time
            db.session.commit()
            flash(f"Güle güle {name}, çıkış kaydedildi.")
        else:
            flash(f"{name} için açık giriş kaydı bulunamadı.")

    return redirect(url_for('home'))

# ----------------- MAIN -----------------
if __name__ == '__main__':
    # prod: app.run(host="0.0.0.0", port=5000)
    app.run(debug=True)
