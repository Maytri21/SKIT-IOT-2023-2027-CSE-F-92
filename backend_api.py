
from pathlib import Path
import json
from io import BytesIO

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)

# Limit uploads to 10 MB.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "brain_mri_cnn.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "results" / "class_names.json"

IMAGE_SIZE = (224, 224)

model = None
class_names = []

# Load the trained model and class mapping.
try:
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if len(class_names) != model.output_shape[-1]:
        raise ValueError("Class names do not match the model output.")

    print("Brain MRI CNN loaded successfully.")
    print("Classes:", class_names)

except Exception as error:
    print("Model loading failed:", error)


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_ready": model is not None,
        "classes": class_names,
    }), 200


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({
            "error": "model_not_ready",
            "message": "The trained CNN model could not be loaded."
        }), 503

    if "file" not in request.files:
        return jsonify({"error": "missing_file"}), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return jsonify({"error": "empty_filename"}), 400

    image_bytes = uploaded_file.read()

    if not image_bytes:
        return jsonify({"error": "empty_file"}), 400

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in ("JPEG", "PNG", "BMP"):
                return jsonify({
                    "error": "unsupported_format",
                    "message": "Upload a JPEG, PNG, or BMP image."
                }), 400

            image = image.convert("RGB")
            image = image.resize(IMAGE_SIZE)

            image_array = np.asarray(image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "error": "invalid_image",
            "message": "The uploaded file is not a valid image."
        }), 400

    try:
        # The model's Rescaling layer divides pixel values by 255.
        probabilities = model.predict(image_array, verbose=0)[0]

        predicted_index = int(np.argmax(probabilities))
        predicted_label = class_names[predicted_index]
        confidence = float(probabilities[predicted_index])

        return jsonify({
            "label": predicted_label,
            "confidence": confidence,
            "class_probabilities": {
                name: float(probabilities[index])
                for index, name in enumerate(class_names)
            },
            "message": (
                "Educational image classification only; "
                "not a medical diagnosis."
            ),
        }), 200

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "prediction_failed",
            "message": "The model could not process this image."
        }), 500


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "error": "file_too_large",
        "message": "The maximum upload size is 10 MB."
    }), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
