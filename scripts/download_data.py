"""
UCI SECOM 반도체 데이터셋 다운로드 스크립트
출처: https://archive.ics.uci.edu/dataset/179/secom
"""

import urllib.request
import os
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

URLS = {
    "secom.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data",
    "secom_labels.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data",
}


def download(filename: str, url: str) -> None:
    dest = RAW_DIR / filename
    if dest.exists():
        print(f"[SKIP] {filename} already exists ({dest.stat().st_size:,} bytes)")
        return
    print(f"[DOWN] {filename} ← {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"[DONE] {filename} saved ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    for name, url in URLS.items():
        download(name, url)
    print("\nAll files ready in:", RAW_DIR)
