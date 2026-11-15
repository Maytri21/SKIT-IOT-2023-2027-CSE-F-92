
from pathlib import Path
import json
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent.parent
ERROR_FILE = PROJECT_DIR / "results" / "misclassified_images.json"

with open(ERROR_FILE, "r", encoding="utf-8") as file:
    errors = json.load(file)

error_counts = Counter(
    (item["actual"], item["predicted"])
    for item in errors
)

print("\nMISCLASSIFICATION ANALYSIS")
print("-" * 45)

for (actual, predicted), count in error_counts.most_common():
    print(f"{actual:15} -> {predicted:15}: {count}")

print("\nTotal misclassified images:", len(errors))
