from flask import Flask, render_template_string, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pickle, base64, os
from io import BytesIO
from PIL import Image
import numpy as np
import face_recognition

# --- DB URL (Heroku + local fallback) ---
db_url = os.environ.get("DATABASE_URL")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
if not db_url:
    db_url = "sqlite:///app.db"  # add-on yoksa SQLite kullan

# --- Flask app ---
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "yoklama123")
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# --- DB ---
db = SQLAlchemy(app)

# Uygulama import edilirken DB yaratma = kötü fikir; crash sebebi olur.
# Bunu kaldır:
# with app.app_context():
#     db.create_all()

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.String(32))
    name = db.Column(db.String(100))
    entry_time = db.Column(db.DateTime)
    exit_time = db.Column(db.DateTime)
    duration = db.Column(db.Interval)

with app.app_context():
    db.create_all()

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

  /* Kamera alanı: yan yana yerleşim + responsive */
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

      <!-- Giriş/Çıkış büyük butonlar -->
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

      <!-- Kamera Alanı (YAN YANA) -->
      <div id="cameraArea" class="container" style="display:none; margin-top:30px;">
        <div class="row cam-row justify-content-center align-items-start">
          <!-- Sol: Canlı Kamera -->
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

          <!-- Sağ: Önizleme -->
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
          canvas.width = video.videoWidth || 960;   // 16:9
          canvas.height = video.videoHeight || 540; // 16:9
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
        img = face_recognition.load_image_file(path)
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

# ----------------- FOTOĞRAF İŞLEME -----------------
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
        # 2 saat içinde aynı işlem tekrarı engeli
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
    else:
        flash("Yüz tanınamadı.")
    return redirect(url_for('home'))

# ----------------- MAIN -----------------
if __name__ == '__main__':
    # prod: app.run(host="0.0.0.0", port=5000)
    app.run(debug=True)
