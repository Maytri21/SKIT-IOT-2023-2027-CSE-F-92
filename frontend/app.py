"""ChestScan: a Streamlit frontend for a student chest X-ray CNN project."""
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

st.set_page_config(page_title="ChestScan · X-ray analysis", page_icon="🩻", layout="wide")
endpoint = os.environ.get("PREDICTION_API_URL", "").strip()
demo = not endpoint

st.html("""<style>
.stMainBlockContainer {max-width:1200px;padding-top:2.5rem;}
h1 {letter-spacing:-1.7px;font-weight:750!important;}
h2,h3 {letter-spacing:-.5px;}
[data-testid="stSidebar"] {border-right:1px solid #DFE8ED;}
[data-testid="stMetricValue"] {color:#087F8C;}
.eyebrow {color:#087F8C;font-size:12px;font-weight:700;letter-spacing:2px;margin-bottom:12px;}
.hero-copy {font-size:17px;color:#607582;max-width:630px;line-height:1.6;margin-bottom:28px;}
.brand {font-size:26px;font-weight:750;letter-spacing:-1px;margin-bottom:0;}
.brand-sub {color:#607582;font-size:12px;margin-bottom:36px;}
.empty {border:1px dashed #CCDCE3;border-radius:14px;background:#F8FBFC;padding:46px 20px;text-align:center;color:#607582;}
.empty strong {display:block;font-size:18px;color:#172D3A;margin-bottom:10px;}
.foot {font-size:12px;color:#607582;margin-top:24px;}
</style>""")

with st.sidebar:
    st.html('<p class="brand">✚ ChestScan</p><p class="brand-sub">MEDICAL IMAGING · STUDENT PROJECT</p>')
    page = st.radio("Workspace", ["Analyze X-ray", "Session history", "Project guide"])
    st.divider()
    st.caption("PREDICTION CONNECTION")
    st.badge("Demo mode" if demo else "API configured", icon="🧪" if demo else "🔗")
    st.caption("Simulated results for interface demonstration." if demo else "Requests will be sent to your configured backend.")
    st.divider()
    st.caption("Ayesha · AI/ML\n\nMaytri · Backend\n\nDevansh · Data & Testing\n\nGaurav · Frontend")

st.session_state.setdefault("history", [])

def decode_image(data):
    if len(data) > 10 * 1024 * 1024:
        raise ValueError("Choose an image smaller than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as original:
                if original.format not in ("JPEG", "PNG"):
                    raise ValueError("Only genuine JPEG and PNG images are supported.")
                if original.width < 32 or original.height < 32:
                    raise ValueError("The image is too small. Use an image at least 32 × 32 pixels.")
                original.load()
                return ImageOps.exif_transpose(original).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("This image cannot be opened safely. Choose a valid JPEG or PNG file.") from exc

if page == "Analyze X-ray":
    st.html('<div class="eyebrow">IMAGE CLASSIFICATION WORKSPACE</div>')
    st.title("A clearer view of your X-ray.")
    st.html('<div class="hero-copy">Upload a chest X-ray, review the image, and explore your CNN’s classification in one simple workflow.</div>')
    if demo:
        st.info("Demo mode: results are simulated. No trained model is connected.", icon="🧪")
    st.caption("Educational research prototype. Predictions are not a medical diagnosis.")
    left, right = st.columns([1.1, 1], gap="large")
    with left, st.container(border=True):
        st.subheader("01 / Upload & review")
        st.caption("Use a chest X-ray without patient names or identifying details.")
        uploaded = st.file_uploader("Chest X-ray image", type=["jpg", "jpeg", "png"], max_upload_size=10, help="JPEG or PNG, up to 10 MB. DICOM is not supported.")
        data = None
        preview = None
        digest = None
        if uploaded:
            data = uploaded.getvalue()
            digest = hashlib.sha256(data).hexdigest()
            try:
                preview = decode_image(data)
                st.image(preview, caption=uploaded.name, width="stretch")
                st.caption(f"{preview.width} × {preview.height} pixels · {len(data) / 1024:.0f} KB")
                st.caption("File validation checks readability; it cannot confirm that an image is a chest X-ray.")
            except ValueError as exc:
                st.error(str(exc))
        else:
            st.html('<div class="empty"><strong>Your image preview appears here</strong>JPEG or PNG · up to 10 MB</div>')
        source = "demo" if demo else endpoint
        current_key = (digest, source)
        if st.session_state.get("result_key") != current_key:
            st.session_state.pop("result", None)
        run = st.button("Run demo analysis" if demo else "Analyze X-ray", icon="🔍", type="primary", width="stretch", disabled=preview is None)
        if run:
            st.session_state.pop("result", None)
            started = time.perf_counter()
            try:
                with st.spinner("Preparing demonstration…" if demo else "Waiting for the prediction service…"):
                    # Fixed fixture, deliberately unrelated to image contents.
                    result = {"label": "Pneumonia", "confidence": 0.87} if demo else predict(data, uploaded.name, "image/png" if uploaded.name.lower().endswith(".png") else "image/jpeg", endpoint)
                result.update({"filename": uploaded.name, "demo": demo, "elapsed_seconds": round(time.perf_counter() - started, 3), "created_at": datetime.now(timezone.utc).isoformat()})
                st.session_state["result"] = result
                st.session_state["result_key"] = current_key
                st.session_state["history"].insert(0, result.copy())
                st.session_state["history"] = st.session_state["history"][:20]
            except ValueError as exc:
                st.error(str(exc))
    with right, st.container(border=True):
        st.subheader("02 / Classification result")
        result = st.session_state.get("result")
        if result:
            st.badge("SIMULATED RESULT" if result["demo"] else "MODEL OUTPUT", icon="🧪" if result["demo"] else "📊")
            st.metric("Demonstration class" if demo else "Predicted class", result["label"])
            st.metric("Simulated model score" if demo else "Model confidence score", f'{result["confidence"]:.1%}')
            st.progress(result["confidence"])
            st.caption("This score is a fixed demo value, independent of the uploaded image." if demo else "Confidence is the model’s reported score; it is not a clinical probability or a measure of diagnostic accuracy.")
            st.divider()
            st.write("**Review the output with your project team.**")
            st.write("Use this workspace to demonstrate your image-to-prediction pipeline and inspect model outputs.")
            st.download_button("Download result · JSON", json.dumps(result, indent=2), file_name="chestscan-result.json", mime="application/json", width="stretch")
        else:
            st.html('<div class="empty"><strong>Ready when you are</strong>Upload an image and start analysis to see the class and confidence score.</div>')
        with st.expander("How the workflow works"):
            st.write("1. Upload and inspect a JPEG or PNG.\n2. Start analysis.\n3. The backend preprocesses the image and runs the CNN.\n4. Review the returned class and confidence score.")
            st.caption("In demo mode, step 3 uses a fixed sample response.")
elif page == "Session history":
    st.html('<div class="eyebrow">YOUR WORKSPACE</div>')
    st.title("Session history")
    st.write("Review the latest 20 results from this session. Images are not saved to disk by this frontend.")
    if st.session_state["history"]:
        rows = [{"File": r["filename"], "Class": r["label"], "Score": f'{r["confidence"]:.1%}', "Mode": "Simulated" if r["demo"] else "API", "Time (UTC)": r["created_at"]} for r in st.session_state["history"]]
        st.dataframe(rows, hide_index=True, width="stretch")
        st.download_button("Export session · JSON", json.dumps(st.session_state["history"], indent=2), "chestscan-session.json", "application/json")
        if st.button("Clear session history"):
            st.session_state["history"] = []
            st.rerun()
    else:
        st.info("No results yet. Analyze an image to start your session history.")
else:
    st.html('<div class="eyebrow">ABOUT THE PROJECT</div>')
    st.title("From image to insight.")
    st.write("A student project exploring CNN-based classification of chest X-rays into Normal and Pneumonia classes.")
    a, b, c = st.columns(3)
    with a, st.container(border=True):
        st.subheader("1. Prepare")
        st.write("Collect, clean, split, and preprocess the dataset. Verify class labels and avoid data leakage.")
    with b, st.container(border=True):
        st.subheader("2. Predict")
        st.write("Train and evaluate the CNN. The backend applies the same preprocessing before inference.")
    with c, st.container(border=True):
        st.subheader("3. Review")
        st.write("Upload an image, inspect the output, and export the result for project discussion.")
    st.subheader("Frontend demonstration checklist")
    st.markdown("- Upload a valid JPEG or PNG and inspect its preview.\n- Run analysis and explain the class and score.\n- Show validation with a corrupt image.\n- Download a result and review session history.\n- Explain the backend connection and loading/error states.")
    st.info("No model performance metrics are shown until your team supplies measured evaluation results.")

st.html('<div class="foot">ChestScan · CNN research project · Educational use only</div>')
