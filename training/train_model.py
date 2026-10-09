
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DATASET_DIR = Path(
    r"C:\Users\Maytri\Desktop\final year project\cleaned_brain_mri_dataset"
)

TRAIN_DIR = DATASET_DIR / "Training"
TEST_DIR = DATASET_DIR / "Testing"

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_PATH = MODEL_DIR / "brain_mri_cnn.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

# --------------------------------------------------
# 2. CHECK DATASET
# --------------------------------------------------

if not TRAIN_DIR.is_dir():
    raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

if not TEST_DIR.is_dir():
    raise FileNotFoundError(f"Testing folder not found: {TEST_DIR}")

print("TensorFlow version:", tf.__version__)
print("Training directory:", TRAIN_DIR)
print("Testing directory:", TEST_DIR)

# Create training and validation datasets.
# 20% of Training is reserved for validation.
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)

if num_classes != 4:
    raise ValueError(
        f"Expected 4 class folders, but found {num_classes}: "
        f"{class_names}"
    )

# Ensure Testing has the same class folders and label order.
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("\nClass names and numeric labels:")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")

with open(RESULTS_DIR / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=4)

# Improve input-pipeline performance.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# --------------------------------------------------
# 3. BUILD BASIC CNN
# --------------------------------------------------

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.03),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.RandomTranslation(0.03, 0.03),
    ],
    name="data_augmentation",
)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(224, 224, 3)),
        data_augmentation,
        tf.keras.layers.Rescaling(1.0 / 255),

        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\nModel architecture:")
model.summary()

# --------------------------------------------------
# 4. TRAIN THE MODEL
# --------------------------------------------------

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]

print("\nStarting CNN training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Save the final in-memory model as well.
model.save(MODEL_PATH)

# Save training history.
history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(RESULTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
    json.dump(history_data, f, indent=4)

# --------------------------------------------------
# 5. PLOT ACCURACY AND LOSS
# --------------------------------------------------

epochs_ran = range(1, len(history.history["accuracy"]) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_ran, history.history["accuracy"], label="Training accuracy")
plt.plot(
    epochs_ran,
    history.history["val_accuracy"],
    label="Validation accuracy",
)
plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_plot.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs_ran, history.history["loss"], label="Training loss")
plt.plot(epochs_ran, history.history["val_loss"], label="Validation loss")
plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "loss_plot.png", dpi=150)
plt.close()

# --------------------------------------------------
# 6. EVALUATE ON UNSEEN TEST DATA
# --------------------------------------------------

print("\nEvaluating on the Testing folder...")

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

true_labels = np.concatenate(
    [labels.numpy() for _, labels in test_ds]
)

probabilities = model.predict(test_ds, verbose=1)
predicted_labels = np.argmax(probabilities, axis=1)

report = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(num_classes)),
    target_names=class_names,
    digits=4,
    zero_division=0,
)

matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(num_classes)),
)

print("\n---------------- TEST RESULTS ----------------")
print(f"Test loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print("\nClassification report:")
print(report)
print("Confusion matrix:")
print(matrix)

with open(
    RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8"
) as f:
    f.write(f"Test loss: {test_loss:.4f}\n")
    f.write(f"Test accuracy: {test_accuracy * 100:.2f}%\n\n")
    f.write(report)
    f.write("\nConfusion matrix:\n")
    f.write(np.array2string(matrix))

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names,
)
display.plot(xticks_rotation=45, values_format="d")
plt.title("Brain MRI CNN - Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# Save summary metrics.
metrics = {
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "class_names": class_names,
}

with open(RESULTS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("\nTraining and evaluation completed.")
print("Model saved to:", MODEL_PATH)
print("Results saved to:", RESULTS_DIR)
