
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_DIR / "dataset" / "Testing"
MODEL_PATH = PROJECT_DIR / "models" / "brain_mri_mobilenet.keras"
RESULTS_DIR = PROJECT_DIR / "results" / "mobilenet"

with open(RESULTS_DIR / "class_names.json", "r") as f:
    class_names = json.load(f)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(224, 224),
    batch_size=16,
    class_names=class_names,
    shuffle=False,
)

model = tf.keras.models.load_model(MODEL_PATH)

print("\nEvaluating MobileNetV2 on unseen test data...")

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    digits=4,
    zero_division=0,
)

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
)

print("\nTEST ACCURACY:", round(test_accuracy * 100, 2), "%")
print("TEST LOSS:", round(float(test_loss), 4))
print("\nCLASSIFICATION REPORT:\n")
print(report)
print("\nCONFUSION MATRIX:\n")
print(matrix)

# Save report
(RESULTS_DIR / "test_classification_report.txt").write_text(
    report,
    encoding="utf-8"
)

# Save confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names
)

disp.plot(xticks_rotation=45, values_format="d")
plt.title("MobileNetV2 - Test Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "test_confusion_matrix.png", dpi=150)
plt.close()

# Save metrics
metrics = {
    "test_accuracy": float(test_accuracy),
    "test_loss": float(test_loss),
    "total_test_images": int(len(y_true)),
    "correct_predictions": int(np.sum(y_true == y_pred)),
    "incorrect_predictions": int(np.sum(y_true != y_pred)),
}

with open(RESULTS_DIR / "test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nEvaluation completed!")
print("Results saved in:", RESULTS_DIR)
