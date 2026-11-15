
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

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_DIR / "dataset" / "Training"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results" / "mobilenet"

MODEL_PATH = MODEL_DIR / "brain_mri_mobilenet.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    label_mode="int",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    shuffle=False,
    label_mode="int",
)

class_names = train_ds.class_names
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

print("Classes:", class_names)

# --------------------------------------------------
# 3. DATA AUGMENTATION
# --------------------------------------------------

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.03),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomTranslation(0.03, 0.03),
])

# --------------------------------------------------
# 4. LOAD MOBILENETV2
# --------------------------------------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

# Initially freeze pretrained layers.
base_model.trainable = False

# --------------------------------------------------
# 5. BUILD CLASSIFICATION MODEL
# --------------------------------------------------

inputs = tf.keras.Input(shape=(224, 224, 3))

x = augmentation(inputs)

# MobileNetV2 expects inputs in [-1, 1].
x = tf.keras.layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --------------------------------------------------
# 6. CALLBACKS
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
        patience=4,
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

# --------------------------------------------------
# 7. TRAIN MODEL
# --------------------------------------------------

print("\nStarting MobileNetV2 training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Save best model, not the last epoch.
best_model = tf.keras.models.load_model(MODEL_PATH)

# Save training history.
history_data = {
    key: [float(v) for v in values]
    for key, values in history.history.items()
}

with open(RESULTS_DIR / "training_history.json", "w") as f:
    json.dump(history_data, f, indent=4)

# --------------------------------------------------
# 8. PLOT TRAINING HISTORY
# --------------------------------------------------

epochs_ran = range(1, len(history.history["accuracy"]) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_ran, history.history["accuracy"], label="Training")
plt.plot(epochs_ran, history.history["val_accuracy"], label="Validation")
plt.title("MobileNetV2 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "accuracy_plot.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs_ran, history.history["loss"], label="Training")
plt.plot(epochs_ran, history.history["val_loss"], label="Validation")
plt.title("MobileNetV2 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "loss_plot.png", dpi=150)
plt.close()

# --------------------------------------------------
# 9. SAVE CLASS NAMES
# --------------------------------------------------

with open(RESULTS_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f, indent=4)

print("\nMobileNetV2 training completed.")
print("Best model saved:", MODEL_PATH)
print("Results saved:", RESULTS_DIR)
