"""三组训练/测试集比例对比实验 (8:2 / 7:3 / 6:4)。"""
import json
import os
import sys
import time
import traceback

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from core.ceo_qaadam_emstgat_trainer import run_training_pipeline

XLSX_PATH = os.path.join(MODEL_DIR, "数据文件", "测试数据.xlsx")
CSV_PATH = os.path.join(MODEL_DIR, "数据文件", "测试数据.csv")


def ensure_csv() -> str:
    """Current trainer expects CSV; convert xlsx once if CSV is stale."""
    if not os.path.exists(XLSX_PATH):
        raise FileNotFoundError(XLSX_PATH)
    needs_build = (
        not os.path.exists(CSV_PATH)
        or os.path.getmtime(CSV_PATH) < os.path.getmtime(XLSX_PATH)
    )
    if needs_build:
        print(f"[prep] converting {XLSX_PATH} -> {CSV_PATH}", flush=True)
        df = pd.read_excel(XLSX_PATH)
        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    return CSV_PATH


EXPERIMENTS = [
    {"test_size": 0.2, "train_ratio": 80, "test_ratio": 20, "output_dir": "results_testdata_80_20"},
    {"test_size": 0.3, "train_ratio": 70, "test_ratio": 30, "output_dir": "results_testdata_70_30"},
    {"test_size": 0.4, "train_ratio": 60, "test_ratio": 40, "output_dir": "results_testdata_60_40"},
]

SUMMARY_DIR = os.path.join(MODEL_DIR, "results", "metrics", "results_testdata_summary")


def main() -> int:
    try:
        data_path = ensure_csv()
    except FileNotFoundError as exc:
        print(f"[FATAL] dataset not found: {exc}")
        return 1

    if not os.path.exists(data_path):
        print(f"[FATAL] dataset not found: {data_path}")
        return 1

    os.makedirs(SUMMARY_DIR, exist_ok=True)
    summaries = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        out_dir = os.path.join(MODEL_DIR, "results", "metrics", exp["output_dir"])
        print(f"\n[{i}/3] test_size={exp['test_size']} -> {out_dir}", flush=True)
        t0 = time.time()
        try:
            results = run_training_pipeline(
                csv_path=data_path,
                test_size=exp["test_size"],
                epochs=100,
                budget=10,
                seed=42,
                use_gpu=True,
                skip_ceo=True,
                output_dir=out_dir,
            )
        except Exception as e:
            print(f"[ERROR] experiment {exp['train_ratio']}:{exp['test_ratio']} failed: {e}")
            traceback.print_exc()
            return 2

        wall = time.time() - t0
        summaries.append({
            "data_path": os.path.basename(data_path),
            "test_size": exp["test_size"],
            "train_ratio": exp["train_ratio"],
            "test_ratio": exp["test_ratio"],
            "epochs": 100,
            "budget": 10,
            "seed": 42,
            "skip_ceo": True,
            "output_dir": exp["output_dir"],
            "accuracy": results.get("accuracy"),
            "precision": results.get("precision"),
            "recall": results.get("recall"),
            "f1_score": results.get("f1_score"),
            "prediction_time": results.get("prediction_time"),
            "training_time": results.get("training_time"),
            "wall_time": wall,
        })

        with open(os.path.join(SUMMARY_DIR, "split_metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        print(f"[{i}/3] done in {wall:.1f}s, acc={results.get('accuracy'):.5f}", flush=True)

    print("\nAll three experiments finished. Summary:")
    for s in summaries:
        print(f"  {s['train_ratio']}:{s['test_ratio']}  acc={s['accuracy']:.5f}  f1={s['f1_score']:.5f}  train={s['training_time']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
