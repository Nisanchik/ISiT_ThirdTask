import csv
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"
IMAGE_DIR.mkdir(exist_ok=True)

with (ROOT / "samples.csv").open(encoding="utf-8-sig", newline="") as file:
    samples = list(csv.DictReader(file))

session = requests.Session()
session.headers["User-Agent"] = (
    "AircraftClassificationStudy/1.2 (small educational photo dataset)"
)
retry = Retry(
    total=4,
    backoff_factor=8,
    status_forcelist=[429, 500, 502, 503, 504],
    respect_retry_after_header=True,
)
session.mount("https://", HTTPAdapter(max_retries=retry))

missing = [row for row in samples if not (IMAGE_DIR / f"{row['id']}.jpg").is_file()]
photos = {}

if missing:
    try:
        response = session.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "titles": "|".join(row["article"] for row in missing),
                "prop": "pageimages",
                "pithumbsize": 640,
                "redirects": 1,
                "format": "json",
            },
            timeout=40,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise SystemExit(
            f"Википедия временно не отвечает: {error}\n"
            "Скачанные фотографии сохранены. Повторите запуск позже."
        ) from error

    query = response.json()["query"]
    pages = {page["title"]: page for page in query["pages"].values()}
    aliases = {
        item["from"]: item["to"]
        for group in ("normalized", "redirects")
        for item in query.get(group, [])
    }

    for row in missing:
        title = row["article"]
        for _ in range(3):
            title = aliases.get(title, title)
        photos[row["id"]] = pages.get(title, {}).get("thumbnail", {}).get("source")

sources = []
failed = []

for index, row in enumerate(samples, 1):
    path = IMAGE_DIR / f"{row['id']}.jpg"
    article_url = "https://en.wikipedia.org/wiki/" + quote(
        row["article"].replace(" ", "_")
    )
    image_url = photos.get(row["id"], "")

    if path.is_file():
        print(f"{index:02d}/{len(samples)}: {row['article']} — уже скачано")
    elif not image_url:
        print(f"{index:02d}/{len(samples)}: {row['article']} — фото не найдено")
        failed.append(row["article"])
    else:
        try:
            response = session.get(image_url, timeout=40)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image.thumbnail((640, 640))
            image.save(path, "JPEG", quality=85)
            print(f"{index:02d}/{len(samples)}: {row['article']} — скачано")
            time.sleep(3)
        except (requests.RequestException, OSError) as error:
            print(f"{index:02d}/{len(samples)}: ошибка — {error}")
            failed.append(row["article"])

    sources.append({"id": row["id"], "article_url": article_url,
                    "image_url": image_url})

with (ROOT / "sources.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=["id", "article_url", "image_url"])
    writer.writeheader()
    writer.writerows(sources)

ready = len(samples) - len(failed)
print(f"Готово: {ready}/{len(samples)} фотографий в папке images")
if failed:
    print("Не скачались: " + ", ".join(failed))
    print("Повторите запуск позже: готовые фотографии будут пропущены.")
