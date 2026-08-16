import os
import sqlite3
import cv2
import numpy as np
from insightface.app import FaceAnalysis

PHOTO_DIR = "/Users/synyster7x/projects/findmyface/testDataset/images"

conn = sqlite3.connect("faces.db")
cur = conn.cursor()
print(f"Database file created at: {os.getcwd()}")

cur.execute('''
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    x1 INTEGER,
    x2 INTEGER,
    y1 INTEGER,
    y2 INTEGER,
    width INTEGER,
    height INTEGER,
    embeddings BLOB
)
''')
conn.commit()

def process_library():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    for root, _, files in os.walk(PHOTO_DIR):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(root, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            h, w = img.shape[:2]
            faces = app.get(img)
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                emb_bytes = face.normed_embedding.astype(np.float32).tobytes()
                cur.execute('''
                    INSERT INTO files (name, x1, x2, y1, y2, width, height, embeddings)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (path, int(x1), int(x2), int(y1), int(y2), w, h, emb_bytes))
            if faces:
                conn.commit()
                print(f"{path}: {len(faces)} face(s)")
    conn.close()

if __name__ == "__main__":
    process_library()