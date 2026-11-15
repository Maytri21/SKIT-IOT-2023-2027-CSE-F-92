
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_DIR / "dataset" / "Testing"
MODEL_PATH = PROJECT_DIR / "models" / "brain_mri_cnn.keras"
RESULTS_DIR = PROJECT_DIR / "results"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

RESULTS_DIR.mkdir(exist_ok=True)

# Load testing dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Predictions
y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=1)

# Evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
loss = model.evaluate(test_ds, verbose=0)[0]

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")
print("Test Loss:", round(float(loss), 4))

# Classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\nClassification Report:\n")
print(report)

(RESULTS_DIR / "classification_report.txt").write_text(
    report,
    encoding="utf-8"
)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

np.savetxt(
    RESULTS_DIR / "confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)

# Save misclassified image records
file_paths = test_ds.file_paths

misclassified = []

for i, (actual, predicted) in enumerate(zip(y_true, y_pred)):
    if actual != predicted:
        misclassified.append({
            "image": Path(file_paths[i]).name,
            "image_path": file_paths[i],
            "actual": class_names[actual],
            "predicted": class_names[predicted],
            "confidence": float(predictions[i][predicted])
        })

with open(
    RESULTS_DIR / "misclassified_images.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(misclassified, f, indent=4)

# Save metrics
metrics = {
    "test_accuracy": float(accuracy),
    "test_loss": float(loss),
    "total_test_images": int(len(y_true)),
    "correct_predictions": int(np.sum(y_true == y_pred)),
    "incorrect_predictions": int(np.sum(y_true != y_pred))
}

with open(
    RESULTS_DIR / "test_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, indent=4)

print("\nEvaluation completed successfully!")
print("Results saved in:", RESULTS_DIR)
print("Misclassified images:", len(misclassified))
