import base64
import csv
import os
import re
import time
from getpass import getpass
from pathlib import Path

from huggingface_hub import InferenceClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results.csv"
MODELS = {
    "Qwen_30B": "Qwen/Qwen3-VL-30B-A3B-Instruct:deepinfra",
    "Qwen_235B": "Qwen/Qwen3-VL-235B-A22B-Instruct:deepinfra",
}
PROMPT = (
    "Classify the airplane in this photo as military or civilian. "
    "Military includes armed-forces transport, patrol, tankers, "
    "trainers and government VIP aircraft operated by armed forces. "
    "Civilian includes airlines, private, firefighting and agricultural planes. "
    "Answer using exactly one word: military or civilian."
)
FIELDS = ["id", "article", "label", "difficulty"]
for name in MODELS:
    FIELDS.extend([name, f"{name}_seconds"])

source = RESULTS if RESULTS.exists() else ROOT / "samples.csv"
with source.open(encoding="utf-8-sig", newline="") as file:
    samples = list(csv.DictReader(file))

for row in samples:
    if not (ROOT / "images" / f"{row['id']}.jpg").is_file():
        raise FileNotFoundError("Сначала запустите prepare_images.py")

token = os.getenv("HF_TOKEN") or getpass("Hugging Face token: ")
client = InferenceClient(api_key=token)


def save_results():
    with RESULTS.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(samples)


for name, model in MODELS.items():
    for row in samples:
        if row.get(name):
            continue
        image = (ROOT / "images" / f"{row['id']}.jpg").read_bytes()
        image_url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
        start = time.perf_counter()
        answer = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]}],
            max_tokens=15,
            temperature=0,
        ).choices[0].message.content
        match = re.search(r"\b(military|civilian)\b", str(answer).lower())
        row[name] = match.group(1) if match else "unknown"
        row[f"{name}_seconds"] = round(time.perf_counter() - start, 2)
        save_results()
        print(f"{name}: {row['id']}/30 → {row[name]}")

metrics = []
for name in MODELS:
    true = [row["label"] for row in samples]
    predicted = [row[name] for row in samples]
    seconds = [float(row[f"{name}_seconds"]) for row in samples]
    metrics.append({
        "model": name,
        "accuracy": round(accuracy_score(true, predicted), 4),
        "precision": round(precision_score(true, predicted, labels=["civilian", "military"], average="macro", zero_division=0), 4),
        "recall": round(recall_score(true, predicted, labels=["civilian", "military"], average="macro", zero_division=0), 4),
        "f1": round(f1_score(true, predicted, labels=["civilian", "military"], average="macro", zero_division=0), 4),
        "avg_seconds": round(sum(seconds) / len(seconds), 2),
    })

with (ROOT / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
    writer.writeheader()
    writer.writerows(metrics)

for row in metrics:
    print(row)
