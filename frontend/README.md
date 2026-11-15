# ChestScan frontend

A Streamlit interface for your four-person chest X-ray classification project.

## Run locally

Open a terminal in this folder. Python 3.12 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. Without an API URL the app runs in visibly labelled demo mode. The fixed Pneumonia / 87% response demonstrates the interface only and does not analyze the image.

## Connect Maytri's backend

Set the endpoint before launching:

```powershell
$env:PREDICTION_API_URL = "http://127.0.0.1:8000/predict"
.venv\Scripts\python.exe -m streamlit run app.py
```

The frontend sends an HTTP POST with a multipart image field named `file`. Return JSON:

```json
{"label": "Pneumonia", "confidence": 0.87}
```

Allowed labels: `Normal`, `Pneumonia`. Confidence must be a finite number from 0 to 1. Adapt `prediction_client.py` if your backend uses different keys or classes. The backend owns model loading, resizing, normalization, inference, and label mapping. The frontend sends original image bytes. Never silently substitute demo predictions when the API fails.

## Frontend contribution

- Responsive upload and result layout, image preview, loading state.
- JPEG/PNG validation, size and image decoding checks, clear errors.
- API adapter with timeouts and response validation.
- Results export, session history and guide page.
- Stale results cleared when the upload changes.

Image contents are not saved to disk by the frontend. Session metadata includes the uploaded filename. A configured API receives the image; its retention policy belongs to your backend. Use anonymized project images. File checks cannot establish that an upload is a chest X-ray. This educational prototype is not a diagnostic tool.

## Practical demonstration

1. Start the app and explain demo versus connected mode.
2. Upload a chest X-ray and show the preview.
3. Run analysis; explain the loading state, label, and model score.
4. Export JSON and open Session history.
5. Upload a corrupt PNG to show the validation error.
6. Connect the actual backend and demonstrate success and a backend failure.

Only claim model inference and accuracy once your team connects and evaluates the real model.
