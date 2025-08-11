# app.py
from flask import Flask, render_template_string, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pickle, os
from io import BytesIO
from PIL import Image
import numpy as np
import face_recognition

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "yoklama123")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

# DB URL (Heroku + local fallback)
db_url = os.environ.get("DATABASE_URL")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
if not db_url:
    db_url = "sqlite:///app.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
if db_url.startswith("postgresql://"):
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "connect_args": {"sslmode": "require"},
        "pool_pre_ping": True,
    }

db = SQLAlchemy(app)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.String(32))
    name = db.Column(db.String(100))
    entry_time = db.Column(db.DateTime)
    exit_time = db.Column(db.DateTime)
    duration = db.Column(db.Interval)

@app.route("/health")
def health():
    return "OK", 200

@app.route("/initdb")
def initdb():
    with app.app_context():
        db.create_all()
    return "DB OK", 200

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
  .cam-card{ background:#fff; border:1px solid #e6eefc; color:var(--text);
    border-radius:16px; padding:16px; }
  #video, #preview{ border-radius:12px; width:100%; height:auto;
    aspect-ratio:16/9; object-fit:cover; max-height:420px; }
  #flashEffect{ display:none; position:fixed; top:0; left:0; width:100%; height:100%;
    background:white; z-index:9999; }
</style>
"""

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
          <div class="text-center"><div style="font-size:42px;">📥</div><div>GİRİŞ</div></div>
        </button>
        <button class="big-square sq-out border-0" onclick="startCamera('exit')">
          <div class="text-center"><div style="font-size:42px;">📤</div><div>ÇIKIŞ</div></div>
        </button>
      </div>
      <div id="cameraArea" class="container" style="display:none;">
        <div class="row cam-row justify-content-center align-items-start">
          <div class="col-12 col-lg-5">
            <div class="cam-card">
              <video id="video" autoplay playsinline></video>
              <div class="mt-3 d-flex gap-2 justify-content-center">
                <button id="snap" class="btn btn-primary">📸 Fotoğraf Çek & Kaydet</button>
                <form id="photoForm"><input type="hidden" id="currentAction" value="/attendance_photo"></form>
              </div>
            </div>
          </div>
          <div class="col-12 col-lg-5"><div class="cam-card">
            <img id="preview" src="" style="display:none;"></div>
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
    navigator.mediaDevices.getUserMedia({{ video: true }})
      .then(stream => document.getElementById('video').srcObject = stream)
      .catch(err => alert('Kamera erişimi reddedildi: ' + err));
  }}
  document.addEventListener('DOMContentLoaded', () => {{
    const snap = document.getElementById('snap');
    snap.onclick = async function(e) {{
      e.preventDefault();
      const video = document.getElementById('video');
      const canvas = document.createElement('canvas');
      canvas.width = 640; canvas.height = 360;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const flash = document.getElementById("flashEffect");
      flash.style.display = "block";
      setTimeout(() => flash.style.display = "none", 200);
      canvas.toBlob(async (blob) => {{
        const fd = new FormData();
        fd.append('photo', blob, 'frame.jpg');
        const url = document.getElementById('currentAction').value;
        await fetch(url, {{ method: 'POST', body: fd }});
        window.location.href = "/";
      }}, 'image/jpeg', 0.7);
    }};
  }});
  {% with messages = get_flashed_messages() %}
    {% if messages %}
      {% for message in messages %}
        alert("{{ message }}");
      {% endfor %}
    {% endif %}
  {% endwith %}
</script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

DASHBOARD_HTML = f'''...'''  # dashboard kısmını senin önceki koddan aynı şekilde bırakabiliriz
ADD_USER_HTML = f'''...'''  # add_user kısmını da aynı şekilde bırakıyoruz

@app.route('/')
def home():
    return render_template_string(HOME_HTML)

@app.route('/dashboard')
def dashboard():
    data = Attendance.query.order_by(Attendance.entry_time.desc()).all()
    return render_template_string(DASHBOARD_HTML, data=data)

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    # önceki kodun ile aynı
    pass

@app.route('/attendance_photo', methods=['POST'])
def attendance_photo():
    return process_photo(is_entry=True)

@app.route('/exit_photo', methods=['POST'])
def exit_photo():
    return process_photo(is_entry=False)

def process_photo(is_entry: bool):
    file = request.files.get('photo')
    if not file:
        flash("Fotoğraf alınamadı.")
        return redirect(url_for('home'))
    image = Image.open(file.stream).convert("RGB")
    img_array = np.array(image)
    face_locs = face_recognition.face_locations(img_array)
    if not face_locs:
        flash("Yüz algılanamadı.")
        return redirect(url_for('home'))
    face_enc = face_recognition.face_encodings(img_array, face_locs)[0]
    if not os.path.exists("face_db.pickle"):
        flash("Yüz veritabanı yok.")
        return redirect(url_for('home'))
    with open("face_db.pickle", "rb") as f:
        known_encodings, known_names, known_ids = pickle.load(f)
    matches = face_recognition.compare_faces(known_encodings, face_enc, tolerance=0.45)
    if True in matches:
        idx = matches.index(True)
        name = known_names[idx]
        person_id = known_ids[idx]
        now = datetime.now()
        last_record = Attendance.query.filter_by(person_id=person_id).order_by(Attendance.entry_time.desc()).first()
        if is_entry:
            yeni = Attendance(person_id=person_id, name=name, entry_time=now)
            db.session.add(yeni)
            db.session.commit()
            flash(f"{name} giriş için fotoğraf çekildi.")
        else:
            if last_record and last_record.exit_time is None:
                last_record.exit_time = now
                last_record.duration = last_record.exit_time - last_record.entry_time
                db.session.commit()
                flash(f"{name} çıkış için fotoğraf çekildi.")
            else:
                flash(f"{name} için açık giriş kaydı bulunamadı.")
    else:
        flash("Yüz tanınamadı.")
    return redirect(url_for('home'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
