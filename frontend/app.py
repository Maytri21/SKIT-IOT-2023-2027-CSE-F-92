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

st.html("""<style>
.stMainBlockContainer {max-width:1240px;padding:2rem 2.5rem 3rem;}
h1 {letter-spacing:-1.6px;font-weight:750!important;}
h2,h3 {letter-spacing:-.6px;}
[data-testid="stSidebar"] {border-right:1px solid #233D50;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {padding:8px 2px;}
[data-testid="stSidebar"] hr {border-color:#2B4355;}
[data-testid="stMetricValue"] {color:#087F8C;font-weight:700;}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:20px!important;}
[data-testid="stFileUploaderDropzone"] {border:1.5px dashed #A6C8D0;border-radius:14px;padding:24px 16px;}
.stButton button,.stDownloadButton button {min-height:46px;border-radius:10px;font-weight:600;}
.stButton button[kind="primary"] {box-shadow:0 6px 18px #087F8C22;}
button:focus-visible,a:focus-visible {outline:3px solid #29B8C7!important;outline-offset:3px;}
.eyebrow {color:#087F8C;font-size:11px;font-weight:750;letter-spacing:2px;margin-bottom:12px;}
.brand {color:#F5FBFF;font-size:28px;font-weight:750;letter-spacing:-1px;margin-bottom:6px;}
.brand span {background:#26B7BD;color:#102A3C;border-radius:10px;padding:1px 8px;margin-right:7px;}
.brand-sub {color:#A4BDC9;font-size:10px;font-weight:600;letter-spacing:1.5px;margin-bottom:30px;}
.hero {position:relative;overflow:hidden;background:linear-gradient(115deg,#112D42,#164D60);border:1px solid #285B6A;border-radius:24px;padding:34px 38px;margin-bottom:20px;}
.hero .eyebrow {color:#82DEE0;}
.hero h1 {color:#F5FBFF;font-size:clamp(30px,3.4vw,46px);line-height:1.15;margin:0 0 16px;max-width:650px;}
.hero p {color:#C4D9E3;font-size:16px;max-width:590px;line-height:1.65;margin:0;}
.hero-tag {display:inline-block;color:#B4EEF0;border:1px solid #4C7C8A;border-radius:30px;padding:6px 12px;font-size:11px;margin-top:22px;letter-spacing:.4px;}
.hero-art {position:absolute;right:36px;top:35px;width:155px;height:155px;opacity:.5;}
.steps {display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:6px 0 22px;}
.step {display:flex;align-items:center;gap:12px;background:#FFFFFF;border:1px solid #DBE6EC;border-radius:14px;padding:16px;}
.step-number {background:#E7F4F4;color:#087F8C;font-size:13px;font-weight:750;border-radius:10px;padding:9px 12px;}
.step strong {display:block;color:#173448;font-size:13px;}
.step small {display:block;color:#5B7180;font-size:11px;margin-top:3px;}
.empty {border:1px dashed #BDD1DC;border-radius:16px;background:linear-gradient(145deg,#F8FBFD,#EDF5F7);padding:38px 20px;text-align:center;color:#5B7180;line-height:1.6;}
.empty strong {display:block;font-size:17px;color:#173448;margin-bottom:8px;}
.empty-icon {display:grid;place-items:center;margin:0 auto 17px;width:54px;height:54px;border-radius:16px;background:#DFF0F2;color:#087F8C;font-size:25px;}
.empty small {display:block;font-size:12px;margin-top:12px;}
.panel-label {color:#087F8C;font-size:10px;font-weight:750;letter-spacing:1.8px;margin:5px 0 2px;}
.foot {font-size:11px;color:#5B7180;border-top:1px solid #DBE6EC;padding-top:18px;margin-top:30px;display:flex;justify-content:space-between;gap:16px;}
@media(max-width:900px) {.hero-art {display:none;} .stMainBlockContainer {padding:1.5rem;} .hero {padding:26px;} }
@media(max-width:600px) {.steps {grid-template-columns:1fr;} .step {padding:12px;} .foot {flex-direction:column;} }
</style>""")

CLASS_DESCRIPTIONS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary",
}


with st.sidebar:
    st.html(
        '<p class="brand"><span>+</span> Brain MRI</p>'
        '<p class="brand-sub">MEDICAL IMAGING WORKSPACE</p>'
    )

    page = st.radio(
        "Workspace",
        ["Analyze MRI", "Session history", "Project guide"],
    )

    st.divider()
    st.caption("PREDICTION CONNECTION")
    st.badge("API configured", icon=":material/link:", color="primary")
    with st.expander("Connection details"):
        st.code(endpoint, language=None)
    st.caption(
        "Predictions are sent to the configured Flask API. "
        "No simulated predictions are used."
    )

    st.divider()
    with st.expander("Meet the project team"):
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
    st.html("""<section class="hero" aria-label="Brain MRI classification">
        <div class="eyebrow">BRAIN MRI CLASSIFICATION</div>
        <h1>A clearer view.<br>A simpler workflow.</h1>
        <p>Upload a brain MRI and explore your CNN’s classification.<br>
        From image preview to results, everything in one place.</p>
        <span class="hero-tag">CNN RESEARCH PROJECT · FOUR-CLASS CLASSIFICATION</span>
        <svg class="hero-art" aria-hidden="true" viewBox="0 0 180 180" fill="none">
        <rect x="5" y="5" width="170" height="170" rx="32" stroke="#7AD9DE"/>
        <circle cx="90" cy="90" r="65" stroke="#7AD9DE" stroke-dasharray="4 8"/>
        <path d="M90 50C72 35 56 46 57 61C35 64 32 89 45 101C31 121 51 141 68 134C75 150 90 143 90 128V50ZM90 50C108 35 124 46 123 61C145 64 148 89 135 101C149 121 129 141 112 134C105 150 90 143 90 128" stroke="#A2EEF0" stroke-width="4" stroke-linecap="round"/>
        <path d="M57 61C71 59 77 70 70 81M45 101C63 96 72 107 65 120M90 89C73 89 77 102 82 107M123 61C109 59 103 70 110 81M135 101C117 96 108 107 115 120M90 89C107 89 103 102 98 107" stroke="#7AD9DE" stroke-width="3" stroke-linecap="round"/>
        </svg></section>
        <div class="steps" aria-label="Analysis steps">
        <div class="step"><span class="step-number">01</span><div><strong>Upload an MRI</strong><small>Choose a JPEG or PNG</small></div></div>
        <div class="step"><span class="step-number">02</span><div><strong>Review & classify</strong><small>Check the preview, then run</small></div></div>
        <div class="step"><span class="step-number">03</span><div><strong>Explore the result</strong><small>View class scores and export</small></div></div>
        </div>""")
    st.caption("Educational research prototype. Model predictions are not a medical diagnosis.")

    left, right = st.columns([1.1, 1], gap="large")

    with left, st.container(border=True):
        st.html('<div class="panel-label">YOUR IMAGE</div>')
        st.subheader("Upload & review")
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
                '<span class="empty-icon" aria-hidden="true">▧</span>'
                '<strong>Start with a brain MRI</strong>'
                'Your image preview will appear here.'
                '<small>JPEG or PNG · Maximum 10 MB</small>' 
                "</div>"
            )

        current_key = (digest, endpoint)

        if st.session_state.get("result_key") != current_key:
            st.session_state.pop("result", None)

        run = st.button(
            "Classify MRI",
            icon=":material/search:",
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
        st.html('<div class="panel-label">ANALYSIS OUTPUT</div>')
        st.subheader("Your result")
        result = st.session_state.get("result")

        if result:
            label = result["label"]
            confidence = result["confidence"]

            st.badge("CNN model output", icon=":material/analytics:", color="primary")

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
                '<span class="empty-icon" aria-hidden="true">◎</span>'
                '<strong>Ready for your first classification</strong>'
                'Upload an image, review the preview, then select Classify MRI.'
                '<small>Your predicted class and confidence will appear here.</small>' 
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
    '<span><strong>Brain MRI</strong> · Medical imaging workspace</span>'
    '<span>CNN research project · Educational use only</span>' 
    "</div>"
)
