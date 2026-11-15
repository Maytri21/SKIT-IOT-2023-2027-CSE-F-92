# AI-Based Detection of Patient Condition from Medical Images

A deep learning-based Brain MRI Classification System that uses a Convolutional Neural Network (CNN) to classify brain MRI images into four categories: Glioma, Meningioma, No Tumor, and Pituitary.

## Project Overview

This project aims to develop an automated medical image classification prototype using CNN and deep learning techniques. It provides a web interface where users can upload MRI images and view model predictions.

**Project ID:** SKIT/IOT/2023-2027/CSE-F-92

## Key Features

- Brain MRI image upload and preview.
- Image validation and preprocessing.
- CNN-based four-class classification.
- Prediction confidence and class probabilities.
- Flask-based prediction API.
- Streamlit-based user interface.
- Session prediction history.
- JSON result export.
- Input validation and error handling.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| Image Processing | Pillow, NumPy |
| Model Evaluation | Scikit-learn |
| Backend | Flask |
| Frontend | Streamlit |
| Visualization | Matplotlib |

## Dataset Structure

```text
dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## Project Structure

```text
project/
├── dataset/
├── frontend/
│   ├── app.py
│   ├── prediction_client.py
│   └── requirements.txt
├── models/
│   └── brain_mri_cnn.keras
├── results/
├── training/
│   ├── train_model.py
│   └── evaluate_model.py
├── backend_api.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd SKIT-IOT-2023-2027-CSE-F-92
```

### 2. Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install backend and model dependencies

```bash
pip install -r requirements.txt
```

### 4. Install frontend dependencies

```bash
pip install -r frontend/requirements.txt
```

## Running the Application

### Start Flask Backend

From the project root:

```bash
python backend_api.py
```

The backend runs at:

```text
http://127.0.0.1:5000
```

Health check:

```text
http://127.0.0.1:5000/health
```

### Start Streamlit Frontend

Open a second terminal:

```powershell
cd frontend
streamlit run app.py
```

Open the local URL displayed by Streamlit, usually:

```text
http://localhost:8501
```

## Model Evaluation

The current saved model achieved **71.84% test accuracy** on the available testing dataset.

Evaluation outputs are stored in the `results/` directory, including:

- Accuracy and loss plots.
- Classification report.
- Confusion matrix.
- Test metrics.
- Training history.

## Disclaimer

This project is an educational research prototype. Its predictions are not medical diagnoses and must not be used as a substitute for professional medical evaluation.

## Contributors

- Ayesha Agarwal
- Maytri Singh Ahluwalia
- Devansh Sharma
- Gaurav Gupta

**Institution:** Swami Keshvanand Institute of Technology, Management & Gramothan (SKIT), Jaipur.