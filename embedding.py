import face_recognition
import pickle
import os

FACES_DIR = "faces"

known_face_encodings = []
known_face_names = []
known_face_ids = []

person_files = [f for f in os.listdir(FACES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
person_files.sort()

for i, file in enumerate(person_files):
    file_path = os.path.join(FACES_DIR, file)
    name = os.path.splitext(file)[0]
    face_id = str(i+1).zfill(3)
    try:
        image = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            print(f"Face not found: {file}, skipped.")
            continue
        known_face_encodings.append(encodings[0])
        known_face_names.append(name)
        known_face_ids.append(face_id)
        print(f"Added: ID={face_id}, Name={name}")
    except Exception as e:
        print(f"Error ({file}): {e}")

print(f"\nTotal {len(known_face_encodings)} saved successfully.")

with open("face_db.pickle", "wb") as f:
    pickle.dump((known_face_encodings, known_face_names, known_face_ids), f)

print("\nface_db.pickle created with auto ID!")
