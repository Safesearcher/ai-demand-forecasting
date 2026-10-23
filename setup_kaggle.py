import os
import json
from pathlib import Path

# Set environment variables
os.environ["KAGGLE_USERNAME"] = "ayan1090"
os.environ["KAGGLE_KEY"] = "d6a967d52e5a523e09b13d83a1ba0e60"

# Create kaggle.json with proper credentials
kaggle_dir = Path("C:\\Users\\dell\\.kaggle")
kaggle_dir.mkdir(parents=True, exist_ok=True)

kaggle_json = {
    "username": "ayan1090",
    "key": "d6a967d52e5a523e09b13d83a1ba0e60"
}

with open(kaggle_dir / "kaggle.json", "w") as f:
    json.dump(kaggle_json, f)

print(f"Created kaggle.json at: {kaggle_dir / 'kaggle.json'}")
print(f"Username: ayan1090")

# Verify authentication
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("Authentication successful!")

    # Try to list competition files
    files = api.competition_list_files('rossmann-store-sales')
    print(f"Successfully listed files for rossmann-store-sales")
    print(f"Number of files: {len(files)}")
    if hasattr(files, '__iter__'):
        for f in files:
            print(f"  - {f.name}")
except Exception as e:
    print(f"Error: {e}")
