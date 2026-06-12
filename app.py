from flask import Flask, render_template, request, url_for
import os, time, cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = 128
FRAMES = 20

model = load_model("deepfakevideo_model.keras")
print("Model loaded successfully!")

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // FRAMES, 1)

    for i in range(FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    while len(frames) < FRAMES:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3)))

    frames = np.array(frames, dtype=np.float32)
    return np.expand_dims(frames, axis=0)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()

    video = request.files["video"]
    filename = secure_filename(video.filename)

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    video.save(save_path)

    frames = extract_frames(save_path)
    pred = model.predict(frames, verbose=0)[0][0]

    if pred >= 0.5:
        result = "FAKE VIDEO"
        confidence = round(pred * 100, 2)
    else:
        result = "REAL VIDEO"
        confidence = round((1 - pred) * 100, 2)

    video_path = url_for("static", filename=f"uploads/{filename}")

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        video_path=video_path,
        model_name="MobileNetV2 + LSTM",
        frames=20,
        resolution="128 × 128",
        prediction_time=round(time.time() - start, 2)
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))