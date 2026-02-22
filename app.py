from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
from ultralytics import YOLO
import os
import torch
import sqlite3
from datetime import datetime

app = Flask(__name__)

# ---------------- MODEL (CPU ONLY + LIGHTWEIGHT) ----------------
model = YOLO("yolov8n.pt")   # nano model
model.to("cpu")

# ---------------- FOLDER ----------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- DATABASE ----------------
DB_NAME = "database.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            upload_time TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------------- HOME ----------------
@app.route('/')
def home():
    return render_template("index.html")

# ======================================================
# IMAGE UPLOAD + DETECTION (CREATE)
# ======================================================
@app.route('/upload_image', methods=['POST'])
def upload_image():

    if 'image' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['image']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    image = cv2.imread(path)

    # Reduce resolution for low RAM cloud
    image = cv2.resize(image, (256, 256))

    # Very small inference size to prevent memory crash
    with torch.no_grad():
        results = model(image, imgsz=160, conf=0.2)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2)

    output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
    cv2.imwrite(output_path, image)

    # SAVE TO DATABASE
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO uploads (filename, upload_time) VALUES (?, ?)",
        (file.filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

    return jsonify({"url": "/uploads/output.jpg"})

# ======================================================
# READ HISTORY (READ)
# ======================================================
@app.route('/history', methods=['GET'])
def get_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM uploads ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    data = []
    for row in rows:
        data.append({
            "id": row[0],
            "filename": row[1],
            "time": row[2]
        })

    return jsonify(data)

# ======================================================
# UPDATE RECORD (UPDATE)
# ======================================================
@app.route('/update/<int:record_id>', methods=['PUT'])
def update_record(record_id):
    data = request.get_json()
    new_name = data.get("filename")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE uploads SET filename=? WHERE id=?",
        (new_name, record_id)
    )
    conn.commit()
    conn.close()

    return jsonify({"message": "Record updated successfully"})

# ======================================================
# DELETE RECORD (DELETE)
# ======================================================
@app.route('/delete/<int:record_id>', methods=['DELETE'])
def delete_record(record_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("SELECT filename FROM uploads WHERE id=?", (record_id,))
    row = cursor.fetchone()

    if row:
        file_path = os.path.join(UPLOAD_FOLDER, row[0])
        if os.path.exists(file_path):
            os.remove(file_path)

    cursor.execute("DELETE FROM uploads WHERE id=?", (record_id,))
    conn.commit()
    conn.close()

    return jsonify({"message": "Record deleted successfully"})

# ======================================================
# SERVE UPLOADED FILES
# ======================================================
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ======================================================
# FIX ANALYTICS 404 (Dummy Route)
# ======================================================
@app.route('/analytics_data')
def analytics_data():
    return jsonify({"total": 0, "fps": 0})

# ======================================================
# RUN APP
# ======================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)