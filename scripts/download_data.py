"""
Downloads both datasets for ClaimPilot using the Kaggle API.
Requires: pip install kaggle, and a valid ~/.kaggle/access_token or kaggle.json.
Run: python scripts/download_data.py
"""
import os
from kaggle.api.kaggle_api_extended import KaggleApi

IMAGE_DATASET = "prajwalbhamere/car-damage-severity-dataset"
TABULAR_DATASET = "arpan129/insurance-fraud-detection"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGE_DIR = os.path.join(DATA_DIR, "car_damage")
TABULAR_DIR = os.path.join(DATA_DIR, "claims")


def main():
    api = KaggleApi()
    api.authenticate()

    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(TABULAR_DIR, exist_ok=True)

    print(f"Downloading {IMAGE_DATASET} ...")
    api.dataset_download_files(IMAGE_DATASET, path=IMAGE_DIR, unzip=True)
    print(f"Done. Contents of {IMAGE_DIR}:")
    print(os.listdir(IMAGE_DIR))

    print(f"Downloading {TABULAR_DATASET} ...")
    api.dataset_download_files(TABULAR_DATASET, path=TABULAR_DIR, unzip=True)
    print(f"Done. Contents of {TABULAR_DIR}:")
    print(os.listdir(TABULAR_DIR))


if __name__ == "__main__":
    main()
