from pathlib import Path
from datetime import datetime
import argparse
import zipfile
import json
import pandas as pd
import numpy as np


def generate_rossmann_sample(target_dir: Path, n_stores: int = 10, n_days: int = 365):
    """Generate synthetic Rossmann dataset with realistic schema."""
    print(f"Generating synthetic Rossmann sample data in {target_dir}...")

    date_range = pd.date_range(start="2013-01-01", periods=n_days, freq="D")

    train_data = []
    for store in range(1, n_stores + 1):
        for date in date_range:
            is_open = np.random.choice([0, 1], p=[0.1, 0.9])
            sales = 0 if is_open == 0 else np.random.randint(100, 8000)
            customers = 0 if is_open == 0 else np.random.randint(50, 500)
            train_data.append({
                "Store": store,
                "DayOfWeek": date.dayofweek + 1,
                "Date": date.strftime("%Y-%m-%d"),
                "Sales": sales,
                "Customers": customers,
                "Open": is_open,
                "Promo": np.random.choice([0, 1], p=[0.7, 0.3]),
                "StateHoliday": np.random.choice(["0", "a", "b", "c"], p=[0.9, 0.05, 0.03, 0.02]),
                "SchoolHoliday": np.random.choice([0, 1], p=[0.8, 0.2])
            })

    train_df = pd.DataFrame(train_data)
    train_df.to_csv(target_dir / "train.csv", index=False)

    store_data = []
    for store in range(1, n_stores + 1):
        store_data.append({
            "Store": store,
            "StoreType": np.random.choice(["a", "b", "c", "d"]),
            "Assortment": np.random.choice(["a", "b", "c"]),
            "CompetitionDistance": np.random.randint(100, 20000) if np.random.random() > 0.1 else np.nan,
            "CompetitionOpenSinceMonth": np.random.randint(1, 13) if np.random.random() > 0.3 else np.nan,
            "CompetitionOpenSinceYear": np.random.randint(2000, 2015) if np.random.random() > 0.3 else np.nan,
            "Promo2": np.random.choice([0, 1], p=[0.6, 0.4]),
            "Promo2SinceWeek": np.random.randint(1, 53) if np.random.random() > 0.4 else np.nan,
            "Promo2SinceYear": np.random.randint(2010, 2015) if np.random.random() > 0.4 else np.nan,
            "PromoInterval": np.random.choice(["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", "Mar,Jun,Sept,Dec", None], p=[0.2, 0.2, 0.2, 0.4])
        })

    store_df = pd.DataFrame(store_data)
    store_df.to_csv(target_dir / "store.csv", index=False)

    print(f"Generated {len(train_df)} rows in train.csv, {len(store_df)} rows in store.csv")


def generate_store_item_sample(target_dir: Path, n_stores: int = 5, n_items: int = 10, n_days: int = 365):
    """Generate synthetic Store Item Demand dataset with realistic schema."""
    print(f"Generating synthetic Store Item sample data in {target_dir}...")

    date_range = pd.date_range(start="2013-01-01", periods=n_days, freq="D")

    train_data = []
    for store in range(1, n_stores + 1):
        for item in range(1, n_items + 1):
            for date in date_range:
                sales = np.random.randint(10, 200)
                train_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "store": store,
                    "item": item,
                    "sales": sales
                })

    train_df = pd.DataFrame(train_data)
    train_df.to_csv(target_dir / "train.csv", index=False)

    print(f"Generated {len(train_df)} rows in train.csv")


def download_datasets(project_root: Path, raw_data_dir: Path, log_file: Path):
    """Download datasets from Kaggle API."""
    datasets = [
        {
            "competition": "rossmann-store-sales",
            "target_dir": raw_data_dir / "rossmann",
            "expected_files": ["train.csv", "store.csv"]
        },
        {
            "competition": "demand-forecasting-kernels-only",
            "target_dir": raw_data_dir / "store_item",
            "expected_files": ["train.csv"]
        }
    ]

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print("Authentication failed. Please ensure your Kaggle API credentials are configured:")
        print("  - Set KAGGLE_API_TOKEN environment variable, or")
        print("  - Place kaggle.json in ~/.kaggle/")
        print(f"Error: {e}")
        return

    log_entries = []

    for dataset in datasets:
        competition = dataset["competition"]
        target_dir = dataset["target_dir"]
        expected_files = dataset["expected_files"]

        target_dir.mkdir(parents=True, exist_ok=True)

        all_files_exist = all((target_dir / f).exists() for f in expected_files)

        if all_files_exist:
            print(f"Skipping {competition}: files already exist")
            continue

        print(f"Downloading {competition}...")

        try:
            api.competition_download_files(
                competition,
                path=target_dir,
                quiet=False
            )

            zip_file = target_dir / f"{competition}.zip"
            if zip_file.exists():
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                zip_file.unlink()

            file_info = []
            for file_name in expected_files:
                file_path = target_dir / file_name
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    row_count = len(df)
                    columns = ", ".join(df.columns.tolist())
                    file_info.append(f"{file_name}: {row_count} rows, columns: [{columns}]")
                else:
                    file_info.append(f"{file_name}: NOT FOUND")

            timestamp = datetime.now().isoformat()
            log_entry = f"[{timestamp}] Dataset: {competition}, Files: {', '.join(file_info)}"
            log_entries.append(log_entry)

            print(f"Successfully downloaded {competition}")

        except Exception as e:
            error_msg = str(e).lower()
            if "403" in error_msg or "401" in error_msg or "authentication" in error_msg:
                print(f"ERROR: Failed to download {competition}")
                print(f"Please accept the competition rules at: https://www.kaggle.com/c/{competition}/rules")
            else:
                print(f"ERROR downloading {competition}: {e}")

    if log_entries:
        with open(log_file, "a") as f:
            f.write("\n".join(log_entries) + "\n")
        print(f"\nAcquisition log written to: {log_file}")

        # Also write / update acquisition_log.json
        json_log_path = raw_data_dir / "acquisition_log.json"
        existing_json = []
        if json_log_path.exists():
            try:
                with open(json_log_path, "r") as f:
                    existing_json = json.load(f)
            except Exception:
                existing_json = []

        for dataset in datasets:
            comp = dataset["competition"]
            tdir = dataset["target_dir"]
            files_dict = {}
            for fname in dataset["expected_files"]:
                fpath = tdir / fname
                if fpath.exists():
                    tdf = pd.read_csv(fpath)
                    files_dict[fname] = {
                        "rows": len(tdf),
                        "columns": tdf.columns.tolist()
                    }
            if files_dict:
                existing_json.append({
                    "timestamp": datetime.now().isoformat(),
                    "dataset": comp,
                    "source": "Kaggle API",
                    "files": files_dict
                })

        with open(json_log_path, "w") as f:
            json.dump(existing_json, f, indent=2)
        print(f"Acquisition JSON log written to: {json_log_path}")


def main():
    parser = argparse.ArgumentParser(description="Download data from Kaggle or generate synthetic samples")
    parser.add_argument("--generate-sample", action="store_true",
                        help="Generate synthetic sample data for testing without Kaggle credentials")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / "data" / "raw"
    log_file = raw_data_dir / "acquisition_log.txt"
    json_log_path = raw_data_dir / "acquisition_log.json"

    raw_data_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_sample:
        print("Generating synthetic sample data...")
        rossmann_dir = raw_data_dir / "rossmann"
        store_item_dir = raw_data_dir / "store_item"

        rossmann_dir.mkdir(parents=True, exist_ok=True)
        store_item_dir.mkdir(parents=True, exist_ok=True)

        generate_rossmann_sample(rossmann_dir)
        generate_store_item_sample(store_item_dir)

        timestamp = datetime.now().isoformat()
        rossmann_train = pd.read_csv(rossmann_dir / "train.csv")
        rossmann_store = pd.read_csv(rossmann_dir / "store.csv")
        store_item_train = pd.read_csv(store_item_dir / "train.csv")
        log_entries = [
            f"[{timestamp}] Dataset: rossmann-store-sales (synthetic), Files: train.csv: {len(rossmann_train)} rows, columns: [{', '.join(rossmann_train.columns.tolist())}]; store.csv: {len(rossmann_store)} rows, columns: [{', '.join(rossmann_store.columns.tolist())}]",
            f"[{timestamp}] Dataset: demand-forecasting-kernels-only (synthetic), Files: train.csv: {len(store_item_train)} rows, columns: [{', '.join(store_item_train.columns.tolist())}]",
        ]
        with open(log_file, "a") as f:
            f.write("\n".join(log_entries) + "\n")
        print(f"\nAcquisition log written to: {log_file}")

        sample_json = [
            {
                "timestamp": timestamp,
                "dataset": "rossmann-store-sales (synthetic)",
                "source": "Synthetic Generator",
                "files": {
                    "train.csv": {"rows": len(rossmann_train), "columns": rossmann_train.columns.tolist()},
                    "store.csv": {"rows": len(rossmann_store), "columns": rossmann_store.columns.tolist()}
                }
            },
            {
                "timestamp": timestamp,
                "dataset": "demand-forecasting-kernels-only (synthetic)",
                "source": "Synthetic Generator",
                "files": {
                    "train.csv": {"rows": len(store_item_train), "columns": store_item_train.columns.tolist()}
                }
            }
        ]
        with open(json_log_path, "w") as f:
            json.dump(sample_json, f, indent=2)
        print(f"Acquisition JSON log written to: {json_log_path}")
    else:
        download_datasets(project_root, raw_data_dir, log_file)


if __name__ == "__main__":
    main()
