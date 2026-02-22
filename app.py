from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
from ultralytics import YOLO
import os
import time

app = Flask(__name__)

model = YOLO("yolov8n.pt")

camera = cv2.VideoCapture(0)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

object_data = {}
fps_value = 0

@app.route('/')
def home():
    return render_template("index.html")




# ---------------- IMAGE DETECTION ----------------
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['image']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

image = cv2.imread(path)

# Reduce image size for cloud speed
image = cv2.resize(image, (320, 320))

with torch.no_grad():
    results = model(image, imgsz=224, conf=0.2, device="cpu")[0]

    for box in results.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])

        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(image,f"{label} {conf:.2f}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

    output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
    cv2.imwrite(output_path, image)

    return jsonify({"url": "/uploads/output.jpg"})


# ---------------- VIDEO DETECTION ----------------
@app.route('/upload_video', methods=['POST'])
def upload_video():

    if 'video' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['video']

    input_path = os.path.join(UPLOAD_FOLDER, "input.mp4")
    file.save(input_path)

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        return jsonify({"error": "Cannot open video"}), 500

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Fix FPS issue
    if fps == 0 or fps is None:
        fps = 20.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(UPLOAD_FOLDER, "output.mp4")

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=320)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame,
                        f"{label} {conf:.2f}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,255),2)

        out.write(frame)

    cap.release()
    out.release()

    return jsonify({"url": "/uploads/output.mp4"})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/analytics_data')
def analytics_data():
    return jsonify({
        "objects": object_data,
        "fps": fps_value,
        "total": sum(object_data.values())
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)