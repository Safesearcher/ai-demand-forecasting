"""
run_all.py
==========
Execute all data pipeline scripts in the correct order.

Usage:
    python run_all.py                    # Download real Kaggle data + run all
    python run_all.py --sample           # Use synthetic sample data (no Kaggle needed)
    python run_all.py --skip-download    # Skip download (use existing raw files)

Prerequisites:
    pip install -r requirements.txt
    Configure Kaggle credentials (see README.md)
"""

import argparse
import subprocess
import sys
from pathlib import Path


STEPS = [
    ("01_download_data.py",       "Step 1: Data Acquisition"),
    ("02_clean_and_validate.py",  "Step 2: Cleaning & Validation"),
    ("03_feature_engineering.py", "Step 3: Feature Engineering"),
    ("04_eda_visualizations.py",  "Step 4: EDA Visualizations"),
]


def run_step(script: str, label: str, extra_args: list = None):
    scripts_dir = Path(__file__).parent / "data_scripts"
    cmd = [sys.executable, str(scripts_dir / script)] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Running: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}. Stopping.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run all data pipeline steps")
    parser.add_argument("--sample", action="store_true",
                        help="Use synthetic sample data (no Kaggle credentials needed)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip the download step (use existing raw files)")
    args = parser.parse_args()

    for script, label in STEPS:
        if script == "01_download_data.py":
            if args.skip_download:
                print("\nSkipping download step (--skip-download)")
                continue
            extra = ["--generate-sample"] if args.sample else []
            run_step(script, label, extra)
        else:
            run_step(script, label)

    print("\n" + "=" * 60)
    print("  All pipeline steps completed successfully.")
    print("  Outputs:")
    print("    data/processed/rossmann_clean.parquet")
    print("    data/processed/store_item_clean.parquet")
    print("    data/processed/cleaning_report.json")
    print("    data/processed/rossmann_train_features.parquet")
    print("    data/processed/rossmann_val_features.parquet")
    print("    data/processed/store_item_train_features.parquet")
    print("    data/processed/store_item_val_features.parquet")
    print("    data/processed/feature_config.json")
    print("    report/figures/*.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
