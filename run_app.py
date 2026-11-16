"""Application runner for AI Demand Forecasting & Inventory Decision Support.
Checks environment, starts FastAPI backend, and provides quick URL guidance.
"""

import sys
import subprocess
import shutil
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    print("=" * 70)
    print(" AI Demand Forecasting & Inventory Decision Support")
    print(" Deliverable 3: Live Application Launcher")
    print("=" * 70)

    # 1. Check Python dependencies
    try:
        import fastapi
        import uvicorn
        import lightgbm
        import torch
        print("[OK] Backend dependencies satisfied (FastAPI, LightGBM, PyTorch).")
    except ImportError as e:
        print(f"[ERROR] Missing backend dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

    # 2. Check models directory
    models_dir = root / "models"
    required_models = ["b_lgbm_p50.txt", "a_lgbm_p50.txt"]
    for m in required_models:
        if (models_dir / m).exists():
            print(f"[OK] Model weight found: {m}")
        else:
            print(f"[WARNING] Model weight not found: {m} (fallback logic will be used)")

    # 3. Inform launch command
    print("\nStarting FastAPI backend server on http://127.0.0.1:8000 ...")
    print("OpenAPI Swagger Documentation: http://127.0.0.1:8000/docs")
    print("React Frontend (Vite): run 'npm run dev' inside 'frontend/' folder (http://localhost:5173)")
    print("-" * 70)

    subprocess.run([
        sys.executable, "-m", "uvicorn", "src.api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ], cwd=root)


if __name__ == "__main__":
    main()
