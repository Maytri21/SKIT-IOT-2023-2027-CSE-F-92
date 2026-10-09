"""Frontend adapter. Configure PREDICTION_API_URL to connect your backend."""
import math
import requests


def validate_result(payload):
    if not isinstance(payload, dict):
        raise ValueError("The prediction service returned an invalid response.")
    label = payload.get("label")
    if label not in ("Normal", "Pneumonia"):
        raise ValueError("The prediction service returned an unsupported class.")
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("The prediction service must return a numeric confidence.")
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1.")
    return {"label": label, "confidence": float(confidence)}


def predict(image_bytes, filename, mime_type, endpoint):
    try:
        response = requests.post(
            endpoint,
            files={"file": (filename, image_bytes, mime_type)},
            timeout=(10, 60),
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise ValueError("The prediction service timed out. Please try again.") from exc
    except requests.RequestException as exc:
        raise ValueError("Cannot reach the prediction service. Check that the backend is running.") from exc
    try:
        return validate_result(response.json())
    except requests.exceptions.JSONDecodeError as exc:
        raise ValueError("The prediction service did not return valid JSON.") from exc
