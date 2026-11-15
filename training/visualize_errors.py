
from pathlib import Path
import json
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent

ERROR_FILE = PROJECT_DIR / "results" / "misclassified_images.json"
OUTPUT_FILE = PROJECT_DIR / "results" / "meningioma_errors.png"

with open(ERROR_FILE, "r", encoding="utf-8") as f:
    errors = json.load(f)

# Select meningioma images predicted as notumor
selected = [
    item for item in errors
    if item["actual"] == "meningioma"
    and item["predicted"] == "notumor"
][:12]

if not selected:
    print("No matching images found.")
    raise SystemExit

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for ax, item in zip(axes, selected):
    image_path = Path(item["image_path"])

    try:
        img = Image.open(image_path).convert("RGB")
        ax.imshow(img)
        ax.set_title(
            f"Actual: {item['actual']}\n"
            f"Predicted: {item['predicted']}\n"
            f"Confidence: {item['confidence']:.2%}",
            fontsize=9
        )
    except Exception as e:
        ax.set_title(f"Could not load image\n{e}")

    ax.axis("off")

for ax in axes[len(selected):]:
    ax.axis("off")

plt.suptitle(
    "Meningioma Images Misclassified as No Tumor",
    fontsize=15
)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150)
plt.close()

print("Contact sheet saved:", OUTPUT_FILE)
