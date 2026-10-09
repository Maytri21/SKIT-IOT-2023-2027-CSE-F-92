
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from io import BytesIO

app = Flask(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_ready": False,
        "message": "Backend running; CNN model is pending."
    }), 200


@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "missing_file"}), 400

    image_bytes = request.files["file"].read(MAX_FILE_SIZE + 1)

    if not image_bytes:
        return jsonify({"error": "empty_file"}), 400

    if len(image_bytes) > MAX_FILE_SIZE:
        return jsonify({"error": "file_too_large"}), 413

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in ("JPEG", "PNG"):
                return jsonify({"error": "unsupported_format"}), 400
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({"error": "invalid_image"}), 400

    # The trained CNN is not available yet.
    return jsonify({
        "error": "model_not_ready",
        "message": "The image was received, but the CNN is not connected yet."
    }), 503


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
