"""Re-run the original main EMSTGAT experiment with the recorded protocol."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core.ceo_qaadam_emstgat_trainer import run_training_pipeline


def prepare_input(source: Path, output: Path) -> Path:
    """Convert legacy input to numeric features while dropping metadata columns."""
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_excel(source) if source.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(source)
    frame = frame.drop(columns=[name for name in ("测试时间", "State", "state", "tsec") if name in frame.columns])
    label = "类型" if "类型" in frame.columns else ("label" if "label" in frame.columns else frame.columns[-1])
    feature_columns = [name for name in frame.columns if name != label]
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
    prepared = pd.concat([numeric, frame[[label]].reset_index(drop=True)], axis=1)
    csv_path = output / "input.csv"
    prepared.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.data)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = prepare_input(source, output)

    result = run_training_pipeline(
        csv_path=str(csv_path),
        test_size=args.test_size,
        epochs=args.epochs,
        budget=10,
        seed=args.seed,
        use_gpu=True,
        skip_ceo=True,
        output_dir=str(output),
    )
    print({"output_dir": str(output), **result})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
