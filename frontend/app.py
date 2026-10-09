
"""Brain MRI Classification: Streamlit frontend for a student CNN project."""

import hashlib
import io
import json
import os
import time
import warnings
from datetime import datetime, timezone

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
from prediction_client import predict

st.set_page_config(
    page_title="Brain MRI Classification",
    page_icon="🧠",
    layout="wide",
)

endpoint = os.environ.get(
    "PREDICTION_API_URL",
    "http://127.0.0.1:5000/predict",
).strip()

st.html("""
<style>
.stMainBlockContainer {
    max-width: 1200px;
    padding-top: 2.5rem;
}
h1 {
    letter-spacing: -1.7px;
    font-weight: 750 !important;
}
h2, h3 {
    letter-spacing: -.5px;
}
[data-testid="stSidebar"] {
    border-right: 1px solid #DFE8ED;
}
[data-testid="stMetricValue"] {
    color: #087F8C;
}
.eyebrow {
    color: #087F8C;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 12px;
}
.hero-copy {
    font-size: 17px;
    color: #607582;
    max-width: 630px;
    line-height: 1.6;
    margin-bottom: 28px;
}
.brand {
    font-size: 26px;
    font-weight: 750;
    letter-spacing: -1px;
    margin-bottom: 0;
}
.brand-sub {
    color: #607582;
    font-size: 12px;
    margin-bottom: 36px;
}
.empty {
    border: 1px dashed #CCDCE3;
    border-radius: 14px;
    background: #F8FBFC;
    padding: 46px 20px;
    text-align: center;
    color: #607582;
}
.empty strong {
    display: block;
    font-size: 18px;
    color: #172D3A;
    margin-bottom: 10px;
}
.foot {
    font-size: 12px;
    color: #607582;
    margin-top: 24px;
}
</style>
""")

CLASS_DESCRIPTIONS = {
    "glioma": "Model-predicted class: Glioma",
    "meningioma": "Model-predicted class: Meningioma",
    "notumor": "Model-predicted class: No Tumor",
    "pituitary": "Model-predicted class: Pituitary",
}


with st.sidebar:
    st.html(
        '<p class="brand">🧠 Brain MRI</p>'
        '<p class="brand-sub">MEDICAL IMAGING · STUDENT PROJECT</p>'
    )

    page = st.radio(
        "Workspace",
        ["Analyze MRI", "Session history", "Project guide"],
    )

    st.divider()
    st.caption("PREDICTION CONNECTION")
    st.code(endpoint, language=None)
    st.caption(
        "Predictions are sent to the configured Flask API. "
        "No simulated predictions are used."
    )

    st.divider()
    st.caption(
        "AI/ML · Model training\n\n"
        "Maytri · Backend integration\n\n"
        "Data & Testing · Evaluation\n\n"
        "UI/Frontend · Streamlit"
    )

st.session_state.setdefault("history", [])


def decode_image(data):
    """Validate and prepare an uploaded JPEG or PNG for display."""

    if len(data) > 10 * 1024 * 1024:
        raise ValueError("Choose an image smaller than 10 MB.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error", Image.DecompressionBombWarning
            )

            with Image.open(io.BytesIO(data)) as original:
                if original.format not in ("JPEG", "PNG"):
                    raise ValueError(
                        "Only genuine JPEG and PNG images are supported."
                    )

                if original.width < 32 or original.height < 32:
                    raise ValueError(
                        "The image is too small. Use an image at least "
                        "32 × 32 pixels."
                    )

                original.load()
                return ImageOps.exif_transpose(original).convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError(
            "This image cannot be opened safely. "
            "Choose a valid JPEG or PNG file."
        ) from exc


if page == "Analyze MRI":
    st.html(
        '<div class="eyebrow">IMAGE CLASSIFICATION WORKSPACE</div>'
    )
    st.title("Brain MRI Classification")
    st.html(
        '<div class="hero-copy">'
        "Upload a brain MRI image, review it, and inspect the "
        "prediction returned by your trained four-class CNN."
        "</div>"
    )

    left, right = st.columns([1.1, 1], gap="large")

    with left, st.container(border=True):
        st.subheader("01 / Upload & review")
        st.caption(
            "Upload a de-identified brain MRI image. "
            "JPEG and PNG are supported; DICOM is not supported."
        )

        uploaded = st.file_uploader(
            "Brain MRI image",
            type=["jpg", "jpeg", "png"],
            max_upload_size=10,
            help="JPEG or PNG, up to 10 MB.",
        )

        data = None
        preview = None
        digest = None

        if uploaded:
            data = uploaded.getvalue()
            digest = hashlib.sha256(data).hexdigest()

            try:
                preview = decode_image(data)
                st.image(
                    preview,
                    caption=uploaded.name,
                    width="stretch",
                )
                st.caption(
                    f"{preview.width} × {preview.height} pixels · "
                    f"{len(data) / 1024:.0f} KB"
                )
                st.caption(
                    "File validation checks readability; it cannot "
                    "confirm that an image is a brain MRI."
                )
            except ValueError as exc:
                st.error(str(exc))

        else:
            st.html(
                '<div class="empty">'
                '<strong>Your MRI preview appears here</strong>'
                "JPEG or PNG · up to 10 MB"
                "</div>"
            )

        current_key = (digest, endpoint)

        if st.session_state.get("result_key") != current_key:
            st.session_state.pop("result", None)

        run = st.button(
            "Classify MRI",
            icon="🔍",
            type="primary",
            width="stretch",
            disabled=preview is None,
        )

        if run:
            st.session_state.pop("result", None)
            started = time.perf_counter()

            mime_type = (
                "image/png"
                if uploaded.name.lower().endswith(".png")
                else "image/jpeg"
            )

            try:
                with st.spinner(
                    "Sending image to the CNN prediction service..."
                ):
                    result = predict(
                        data,
                        uploaded.name,
                        mime_type,
                        endpoint,
                    )

                result.update({
                    "filename": uploaded.name,
                    "elapsed_seconds": round(
                        time.perf_counter() - started, 3
                    ),
                    "created_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                })

                st.session_state["result"] = result
                st.session_state["result_key"] = current_key

                st.session_state["history"].insert(0, result.copy())
                st.session_state["history"] = (
                    st.session_state["history"][:20]
                )

            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))

    with right, st.container(border=True):
        st.subheader("02 / Classification result")
        result = st.session_state.get("result")

        if result:
            label = result["label"]
            confidence = result["confidence"]

            st.badge("CNN MODEL OUTPUT", icon="📊")

            st.metric(
                "Predicted class",
                CLASS_DESCRIPTIONS.get(label, label),
            )
            st.metric(
                "Model confidence score",
                f"{confidence:.1%}",
            )
            st.progress(confidence)

            st.caption(
                "The confidence score is the model's output, not a "
                "clinical probability or proof of diagnostic accuracy."
            )

            st.divider()
            st.write("**Prediction details**")
            st.write(f"Image: `{result['filename']}`")
            st.write(
                f"Processing time: "
                f"{result['elapsed_seconds']:.2f} seconds"
            )

            probabilities = result.get("class_probabilities")
            if isinstance(probabilities, dict):
                st.write("**Scores for all model classes**")
                for class_name, score in probabilities.items():
                    if isinstance(score, (int, float)):
                        st.write(
                            f"{CLASS_DESCRIPTIONS.get(class_name, class_name)}"
                            f": {score:.1%}"
                        )
                        st.progress(max(0.0, min(1.0, float(score))))

            st.info(
                "This result is for educational demonstration only "
                "and must not be interpreted as a diagnosis."
            )

            st.download_button(
                "Download result · JSON",
                json.dumps(result, indent=2),
                file_name="brain-mri-result.json",
                mime="application/json",
                width="stretch",
            )

        else:
            st.html(
                '<div class="empty">'
                '<strong>Ready when you are</strong>'
                "Upload an image and classify it to see the model output."
                "</div>"
            )

        with st.expander("How the workflow works"):
            st.write(
                "1. Upload and inspect a JPEG or PNG image.\n"
                "2. Send the image to the Flask API.\n"
                "3. Flask preprocesses the image and runs the trained CNN.\n"
                "4. Review the predicted class and confidence score."
            )
            st.caption(
                "If the API or model is unavailable, the app displays "
                "an error rather than inventing a prediction."
            )


elif page == "Session history":
    st.html('<div class="eyebrow">YOUR WORKSPACE</div>')
    st.title("Session history")
    st.write(
        "Review the latest 20 predictions from this Streamlit session. "
        "Uploaded images are not saved to disk by this frontend."
    )

    if st.session_state["history"]:
        rows = [
            {
                "File": item["filename"],
                "Class": item["label"],
                "Score": f'{item["confidence"]:.1%}',
                "Time (UTC)": item["created_at"],
            }
            for item in st.session_state["history"]
        ]

        st.dataframe(rows, hide_index=True, width="stretch")

        st.download_button(
            "Export session · JSON",
            json.dumps(st.session_state["history"], indent=2),
            file_name="brain-mri-session.json",
            mime="application/json",
        )

        if st.button("Clear session history"):
            st.session_state["history"] = []
            st.session_state.pop("result", None)
            st.session_state.pop("result_key", None)
            st.rerun()

    else:
        st.info(
            "No predictions yet. Analyze an MRI to start your session history."
        )


else:
    st.html('<div class="eyebrow">ABOUT THE PROJECT</div>')
    st.title("From MRI image to model output.")
    st.write(
        "A student project exploring CNN-based classification of "
        "brain MRI images into four dataset classes."
    )

    a, b, c = st.columns(3)

    with a, st.container(border=True):
        st.subheader("1. Prepare")
        st.write(
            "Clean the dataset, verify class labels, split the images, "
            "and check for duplicate or near-duplicate images."
        )

    with b, st.container(border=True):
        st.subheader("2. Predict")
        st.write(
            "Train and evaluate the CNN. The Flask backend applies "
            "the model's preprocessing before inference."
        )

    with c, st.container(border=True):
        st.subheader("3. Evaluate")
        st.write(
            "Review predictions and measured evaluation metrics. "
            "Do not treat model outputs as medical diagnoses."
        )

    st.subheader("Frontend demonstration checklist")
    st.markdown(
        "- Upload a valid JPEG or PNG and inspect its preview.\n"
        "- Send the image to the running Flask API.\n"
        "- Test error handling with an invalid image.\n"
        "- Download a prediction and review session history.\n"
        "- Explain the model evaluation and limitations."
    )

    st.info(
        "The CNN's measured test accuracy is 71.84% on the current test "
        "set. This result does not establish clinical reliability."
    )

st.html(
    '<div class="foot">'
    "Brain MRI Classification · CNN research project · Educational use only"
    "</div>"
)
