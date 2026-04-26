from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# ==========================
# LOAD MODEL
# ==========================
try:
    model = tf.keras.models.load_model("model.h5")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading error:", e)
    model = None

# ⚠️ Must match training order
LABELS = ['iron', 'magnesium', 'nitrogen', 'phosphorus', 'potassium']


# ==========================
# HOME
# ==========================
@app.route("/")
def index():
    return render_template("index.html")


# ==========================
# PREDICT API
# ==========================
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        # 🔒 Validate file
        if file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        # ==========================
        # IMAGE PREPROCESSING
        # ==========================
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ==========================
        # MODEL PREDICTION
        # ==========================
        preds = model.predict(img_array, verbose=0)[0]

        # 🔥 Debug (optional)
        print("RAW:", preds)

        # ==========================
        # CONVERT TO %
        # ==========================
        scores = {
            LABELS[i]: round(float(preds[i] * 100), 2)
            for i in range(len(LABELS))
        }

        # 🔥 Get top prediction
        top_label = LABELS[np.argmax(preds)]
        top_conf = round(float(np.max(preds) * 100), 2)

        return jsonify({
            "scores": scores,
            "top_prediction": top_label,
            "confidence": top_conf
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": "Prediction failed"}), 500


# ==========================
# RUN APP
# ==========================
if __name__ == "__main__":
    app.run(debug=True)