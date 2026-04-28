"""Download the M5 Forecasting Accuracy dataset from Kaggle.

Requires Kaggle API credentials in ~/.kaggle/kaggle.json.
See https://www.kaggle.com/docs/api for setup instructions.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from forecasting.config import RAW_DATA_DIR

COMPETITION = "m5-forecasting-accuracy"


def download() -> None:
    """Download and extract M5 dataset to data/raw/."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {COMPETITION} to {RAW_DATA_DIR}...")
    try:
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                COMPETITION,
                "-p",
                str(RAW_DATA_DIR),
            ],
            check=True,
        )
    except FileNotFoundError:
        print(
            "ERROR: kaggle CLI not found. Install with: pip install kaggle\n"
            "Then configure credentials at ~/.kaggle/kaggle.json",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Kaggle download failed (exit code {e.returncode})", file=sys.stderr)
        sys.exit(1)

    # Extract zip
    zip_path = RAW_DATA_DIR / f"{COMPETITION}.zip"
    if zip_path.exists():
        print(f"Extracting {zip_path}...")
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DATA_DIR)
        zip_path.unlink()  # Remove zip after extraction

    print(f"Done. Files available in {RAW_DATA_DIR}")
    for f in sorted(Path(RAW_DATA_DIR).glob("*.csv")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    download()
