
import math
import requests


def validate_result(payload):
    if not isinstance(payload, dict):
        raise ValueError(
            "The prediction service returned an invalid response."
        )

    label = payload.get("label")

    # Brain MRI classes returned by the trained CNN.
    supported_classes = {
        "glioma",
        "meningioma",
        "notumor",
        "pituitary",
    }

    if not isinstance(label, str) or label not in supported_classes:
        raise ValueError(
            f"The prediction service returned an unsupported class: {label}"
        )

    confidence = payload.get("confidence")

    if isinstance(confidence, bool) or not isinstance(
        confidence, (int, float)
    ):
        raise ValueError(
            "The prediction service must return a numeric confidence."
        )

    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1.")

    return {
        "label": label,
        "confidence": float(confidence),
    }


def predict(image_bytes, filename, mime_type, endpoint):
    """Send a brain MRI image to the Flask prediction API."""

    try:
        response = requests.post(
            endpoint,
            files={"file": (filename, image_bytes, mime_type)},
            timeout=(10, 120),
        )

    except requests.Timeout as exc:
        raise ValueError(
            "The prediction service timed out. Please try again."
        ) from exc

    except requests.RequestException as exc:
        raise ValueError(
            "Cannot reach the prediction service. "
            "Check that Flask is running."
        ) from exc

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if response.status_code != 200:
        message = payload.get(
            "message",
            payload.get(
                "error",
                f"Prediction API returned HTTP {response.status_code}.",
            ),
        )
        raise ValueError(message)

    return validate_result(payload)
