import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"

# Make sure Python can import from src if needed
sys.path.insert(0, str(SRC_DIR))

# Utils for quick checks
from src.utils import load_data  # this exists and is correct

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("orchestrator")


def run(cmd, cwd=PROJECT_ROOT):
    logger.info(f"▶ Running: {cmd}")
    result = subprocess.run(cmd, cwd=cwd, shell=True)
    if result.returncode != 0:
        raise SystemExit(f"❌ Command failed: {cmd}")


def run_all():
    logger.info("🚀 Starting full federated pipeline...")

    # 0) Ensure cleaned dataset exists (preprocess step should have produced it)
    cleaned = PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv"
    if not cleaned.exists():
        logger.error("Processed dataset not found at data/processed/cleaned_dataset.csv")
        logger.error("Please run: python scripts/preprocess_data.py")
        return
    df = load_data(cleaned)
    logger.info(f"✅ Loaded cleaned dataset ({df.shape[0]} rows, {df.shape[1]} cols)")

    # 1) Split into hospitals
    logger.info("🏥 Splitting dataset into hospital datasets...")
    run(f'python "{SCRIPTS_DIR / "split_federated_data.py"}"')

    # 2) Train local models for each hospital
    logger.info("🧠 Training local models...")
    for i in (1, 2, 3):
        hospital_csv = PROJECT_ROOT / "data" / "federated" / f"hospital_{i}.csv"
        run(f'python "{SCRIPTS_DIR / "train_local_model.py"}" "{hospital_csv}"')

    # 3) Federated aggregation (runs on script execution)
    logger.info("🤝 Aggregating local models (FedAvg)...")
    run(f'python "{SRC_DIR / "federated_averaging.py"}"')

    # 4) Evaluate global vs baselines
    logger.info("📊 Evaluating models on master test set...")
    run(f'python "{SRC_DIR / "evaluation.py"}"')

    logger.info("🎉 Pipeline complete!")


def run_evaluate_only():
    # Only run evaluation step (assumes models already exist)
    run(f'python "{SRC_DIR / "evaluation.py"}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Sepsis Detection Pipeline")
    parser.add_argument("--run", choices=["all", "evaluate"], required=True,
                        help="Which stage to run: all | evaluate")
    args = parser.parse_args()

    if args.run == "all":
        run_all()
    elif args.run == "evaluate":
        run_evaluate_only()