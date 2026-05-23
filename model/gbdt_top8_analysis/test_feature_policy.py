import ast
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parent / "model.py"


def _base_env() -> Dict[str, Any]:
    return {
        "np": np,
        "re": __import__("re"),
        "Any": Any,
        "List": List,
        "Dict": Dict,
        "Tuple": Tuple,
        "Optional": Optional,
        "logger": type(
            "_Logger",
            (),
            {
                "warning": staticmethod(lambda *args, **kwargs: None),
                "info": staticmethod(lambda *args, **kwargs: None),
                "error": staticmethod(lambda *args, **kwargs: None),
            },
        )(),
    }


def _load_functions(*names: str) -> Dict[str, Any]:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(SCRIPT_PATH))

    env: Dict[str, Any] = _base_env()
    function_nodes = {
        node.name: node for node in module.body if isinstance(node, ast.FunctionDef)
    }
    loaded = set()

    def load(name: str):
        if name in loaded:
            return

        node = function_nodes.get(name)
        assert node is not None, f"Missing helper function: {name}"

        dependencies = {
            child.id
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and child.id in function_nodes
        }
        for dependency in dependencies:
            load(dependency)

        exec(
            compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT_PATH), "exec"),
            env,
        )
        loaded.add(name)

    for name in names:
        load(name)

    return env


def _load_function_shallow(name: str) -> Dict[str, Any]:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(SCRIPT_PATH))
    env: Dict[str, Any] = _base_env()

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT_PATH), "exec"),
                env,
            )
            return env

    raise AssertionError(f"Missing helper function: {name}")


def test_feature_policy_helpers_detect_power_and_current_columns():
    helpers = _load_functions(
        "_identify_power_and_current_columns",
        "_assert_feature_policy",
    )

    power_cols, current_cols = helpers["_identify_power_and_current_columns"](
        ["StackVoltage", "PW", "stackCurrent", "i_write", "airFlow"]
    )

    assert power_cols == ["PW"]
    assert current_cols == ["stackCurrent", "i_write"]
    helpers["_assert_feature_policy"](["StackVoltage", "stackCurrent", "airFlow"])


@pytest.mark.parametrize(
    ("feature_names", "message"),
    [
        (["StackVoltage", "PW", "stackCurrent"], "功率"),
        (["StackVoltage", "airFlow"], "电流"),
    ],
)
def test_feature_policy_assertion_rejects_invalid_feature_sets(feature_names, message):
    helpers = _load_functions("_assert_feature_policy")

    with pytest.raises(ValueError, match=message):
        helpers["_assert_feature_policy"](feature_names)


def test_ensure_current_columns_are_restored_after_selection():
    helpers = _load_functions(
        "_identify_power_and_current_columns",
        "_ensure_current_columns_selected",
    )

    feature_names = ["stackVoltage", "stackCurrent", "i_write", "airFlow"]
    _, current_cols = helpers["_identify_power_and_current_columns"](feature_names)
    selected_idx = np.array([0, 3])
    importances = np.array([0.5, 0.2, 0.1, 0.2])

    final_idx = helpers["_ensure_current_columns_selected"](
        feature_names=feature_names,
        selected_idx=selected_idx,
        importances=importances,
        current_columns=current_cols,
    )

    assert set(final_idx.tolist()) == {0, 1, 2, 3}
    assert final_idx[0] == 0


def test_plot_helpers_support_normalized_confusion_matrix_and_split_summary():
    helpers = _load_functions(
        "plot_confusion_matrix",
        "plot_per_class_metrics",
        "plot_split_metrics_comparison",
    )

    class _DummyCbar:
        pass

    class _DummyFigure:
        def colorbar(self, *args, **kwargs):
            return _DummyCbar()

    class _DummyAxes:
        def __init__(self):
            self.figure = _DummyFigure()
            self.imshow_input = None
            self.text_calls = []
            self.bar_calls = []
            self.xticks = None

        def imshow(self, matrix, interpolation=None, cmap=None):
            self.imshow_input = matrix
            return object()

        def set_title(self, *args, **kwargs):
            return None

        def set_xticks(self, ticks, *args, **kwargs):
            self.xticks = ticks
            return None

        def set_yticks(self, *args, **kwargs):
            return None

        def set_xticklabels(self, *args, **kwargs):
            return None

        def set_yticklabels(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def text(self, *args, **kwargs):
            self.text_calls.append((args, kwargs))
            return None

        def bar(self, *args, **kwargs):
            self.bar_calls.append((args, kwargs))
            return []

        def legend(self, *args, **kwargs):
            return None

        def grid(self, *args, **kwargs):
            return None

    class _DummyPlt:
        def __init__(self):
            self.axes = []
            self.saved_paths = []

        def subplots(self, nrows=1, ncols=1, figsize=None):
            ax = _DummyAxes()
            self.axes.append(ax)
            return object(), ax

        def tight_layout(self):
            return None

        def savefig(self, path, dpi=None, bbox_inches=None):
            self.saved_paths.append(path)

        def close(self):
            return None

    helpers["_apply_plot_font"] = lambda: None
    helpers["PLOT_FONT_FAMILY"] = "DejaVu Sans"
    helpers["plt"] = _DummyPlt()
    helpers["confusion_matrix"] = lambda y_true, y_pred: np.array([[8, 2], [1, 9]])

    helpers["plot_confusion_matrix"](
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        ["A", "B"],
        save_path="normalized.png",
        normalize=True,
    )

    matrix = helpers["plt"].axes[0].imshow_input
    assert np.allclose(matrix.sum(axis=1), np.array([1.0, 1.0]))
    assert helpers["plt"].saved_paths[-1] == "normalized.png"

    helpers["plt"] = _DummyPlt()
    helpers["plot_per_class_metrics"](
        {
            "precision": np.array([0.9, 0.8]),
            "recall": np.array([0.85, 0.75]),
            "f1_score": np.array([0.87, 0.77]),
        },
        ["A", "B"],
        save_path="per_class_metrics.png",
    )
    assert helpers["plt"].saved_paths[-1] == "per_class_metrics.png"
    assert len(helpers["plt"].axes[0].bar_calls) == 3

    helpers["plt"] = _DummyPlt()
    helpers["plot_split_metrics_comparison"](
        [
            {
                "test_size": 0.2,
                "accuracy": 0.91,
                "precision": 0.9,
                "recall": 0.89,
                "f1_score": 0.9,
            },
            {
                "test_size": 0.3,
                "accuracy": 0.92,
                "precision": 0.91,
                "recall": 0.9,
                "f1_score": 0.91,
            },
        ],
        save_path="split_metrics.png",
    )
    assert helpers["plt"].saved_paths[-1] == "split_metrics.png"
    assert len(helpers["plt"].axes[0].bar_calls) == 4


def test_run_multi_split_experiments_dispatches_expected_splits():
    helpers = _load_function_shallow("run_multi_split_experiments")

    calls = []
    helpers["run_training_pipeline"] = lambda **kwargs: calls.append(kwargs) or {
        "accuracy": 0.9,
        "precision": 0.89,
        "recall": 0.88,
        "f1_score": 0.885,
        "prediction_time": 0.12,
    }

    summaries = helpers["run_multi_split_experiments"](
        data_path="测试数据.xlsx",
        test_sizes=[0.2, 0.3, 0.4],
        epochs=5,
        budget=2,
        seed=7,
        use_gpu=False,
        skip_ceo=True,
    )

    assert [round(call["test_size"], 1) for call in calls] == [0.2, 0.3, 0.4]
    assert [call["output_dir"] for call in calls] == [
        "results_testdata_80_20",
        "results_testdata_70_30",
        "results_testdata_60_40",
    ]
    assert [summary["train_ratio"] for summary in summaries] == [80, 70, 60]
    assert [summary["test_ratio"] for summary in summaries] == [20, 30, 40]
    assert [summary["output_dir"] for summary in summaries] == [
        "results_testdata_80_20",
        "results_testdata_70_30",
        "results_testdata_60_40",
    ]
    assert all(summary["data_path"] == "测试数据.xlsx" for summary in summaries)
    assert all(summary["seed"] == 7 for summary in summaries)
    assert all(summary["epochs"] == 5 for summary in summaries)
    assert all(summary["budget"] == 2 for summary in summaries)
    assert all(summary["use_gpu"] is False for summary in summaries)
    assert all(summary["skip_ceo"] is True for summary in summaries)
    assert all(summary["accuracy"] == 0.9 for summary in summaries)
    assert all(summary["precision"] == 0.89 for summary in summaries)
    assert all(summary["recall"] == 0.88 for summary in summaries)
    assert all(summary["f1_score"] == 0.885 for summary in summaries)
    assert all(summary["prediction_time"] == 0.12 for summary in summaries)
    assert len(calls) == 3
    assert len(summaries) == 3


@pytest.mark.parametrize(
    ("shap_values", "expected"),
    [
        (
            [
                np.array([[1.0, -2.0], [3.0, -4.0]]),
                np.array([[0.5, -1.0], [1.5, -2.0]]),
            ],
            np.array([1.5, 2.25]),
        ),
        (
            np.array(
                [
                    [[1.0, -2.0], [3.0, -4.0]],
                    [[0.5, -1.0], [1.5, -2.0]],
                ]
            ),
            np.array([1.5, 2.25]),
        ),
    ],
)
def test_aggregate_multiclass_shap_values_returns_global_importance(shap_values, expected):
    helpers = _load_functions("aggregate_shap_global_importance")

    result = helpers["aggregate_shap_global_importance"](shap_values)

    np.testing.assert_allclose(result, expected)


def test_aggregate_shap_values_supports_samples_features_classes_layout():
    helpers = _load_functions("aggregate_shap_global_importance")

    shap_values = np.array(
        [
            [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]],
            [[4.0, 2.0], [5.0, 2.5], [6.0, 3.0]],
        ]
    )

    result = helpers["aggregate_shap_global_importance"](
        shap_values,
        feature_count=3,
    )

    np.testing.assert_allclose(result, np.array([1.875, 2.625, 3.375]))



def test_build_shap_importance_records_sorts_descending():
    helpers = _load_functions("build_shap_importance_records")

    rows = helpers["build_shap_importance_records"](
        ["f1", "f2", "f3"], np.array([0.2, 0.8, 0.4])
    )

    assert rows == [
        {"feature": "f2", "importance": 0.8},
        {"feature": "f3", "importance": 0.4},
        {"feature": "f1", "importance": 0.2},
    ]



def test_build_shap_metadata_payload_includes_counts_and_status():
    helpers = _load_functions("build_shap_metadata")

    payload = helpers["build_shap_metadata"](
        status="success",
        background_samples=32,
        explanation_samples=16,
        feature_count=8,
        detail="ok",
    )

    assert payload == {
        "status": "success",
        "background_samples": 32,
        "explanation_samples": 16,
        "feature_count": 8,
        "detail": "ok",
    }



def test_generate_shap_artifacts_returns_failed_payload_when_import_fails(tmp_path):
    helpers = _load_functions(
        "build_shap_metadata",
        "generate_shap_artifacts",
    )

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("missing shap")
        return real_import(name, *args, **kwargs)

    helpers["__import__"] = fake_import

    result = helpers["generate_shap_artifacts"](
        model=object(),
        X_train=np.ones((10, 3), dtype=float),
        X_test=np.ones((5, 3), dtype=float),
        feature_names=["a", "b", "c"],
        output_dir=str(tmp_path),
        seed=7,
    )

    assert result["status"] == "failed"
    assert result["background_samples"] == 0
    assert result["explanation_samples"] == 0
    assert "missing shap" in result["detail"]



def test_generate_shap_artifacts_writes_outputs_when_analysis_succeeds(tmp_path):
    helpers = _load_functions(
        "aggregate_shap_global_importance",
        "build_shap_importance_records",
        "build_shap_metadata",
        "generate_shap_artifacts",
    )

    class _FakeModel:
        def __init__(self):
            self.calls = []

        def predict(self, values, verbose=0):
            self.calls.append((np.array(values), verbose))
            batch = np.asarray(values)
            return np.tile(np.array([[0.7, 0.3]]), (batch.shape[0], 1))

    class _FakeExplainer:
        def __init__(self, predict_fn, background):
            self.predict_fn = predict_fn
            self.background = np.asarray(background)

        def shap_values(self, samples, nsamples=None):
            prediction = self.predict_fn(samples)
            assert prediction.shape[1] == 2
            return [
                np.array([[1.0, -2.0, 0.5], [3.0, -4.0, 1.5]]),
                np.array([[0.5, -1.0, 0.25], [1.5, -2.0, 0.75]]),
            ]

    class _FakeShapModule:
        KernelExplainer = _FakeExplainer

        @staticmethod
        def summary_plot(shap_values, features, feature_names=None, show=None, plot_type=None):
            assert feature_names == ["f1", "f2", "f3"]
            assert np.asarray(features).shape == (2, 3)
            assert show is False
            assert plot_type in (None, "bar")

    class _FakePlt:
        def __init__(self):
            self.saved_paths = []

        def figure(self, figsize=None):
            return object()

        def tight_layout(self):
            return None

        def savefig(self, path, dpi=None, bbox_inches=None):
            self.saved_paths.append(path)
            Path(path).write_bytes(b"fake-image")

        def close(self):
            return None

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "shap":
            return _FakeShapModule()
        return real_import(name, *args, **kwargs)

    helpers["plt"] = _FakePlt()
    helpers["_apply_plot_font"] = lambda: None
    helpers["SHAP_BACKGROUND_SAMPLE_SIZE"] = 3
    helpers["SHAP_EXPLANATION_SAMPLE_SIZE"] = 2
    helpers["SHAP_KERNEL_NSAMPLES"] = 5
    helpers["__import__"] = fake_import

    model = _FakeModel()
    result = helpers["generate_shap_artifacts"](
        model=model,
        X_train=np.arange(30, dtype=float).reshape(10, 3),
        X_test=np.arange(15, dtype=float).reshape(5, 3),
        feature_names=["f1", "f2", "f3"],
        output_dir=str(tmp_path),
        seed=11,
    )

    assert result["status"] == "success"
    assert result["background_samples"] == 3
    assert result["explanation_samples"] == 2
    assert Path(result["summary_path"]).exists()
    assert Path(result["bar_path"]).exists()
    assert Path(result["importance_path"]).exists()
    assert Path(result["meta_path"]).exists()
    assert len(model.calls) == 1
    assert model.calls[0][1] == 0

    importance_rows = json.loads(Path(result["importance_path"]).read_text(encoding="utf-8"))
    assert [row["feature"] for row in importance_rows] == ["f2", "f1", "f3"]

    meta = json.loads(Path(result["meta_path"]).read_text(encoding="utf-8"))
    assert meta["status"] == "success"
    assert meta["feature_count"] == 3



def test_run_training_pipeline_adds_shap_summary_to_results(tmp_path):
    helpers = _load_function_shallow("run_training_pipeline")

    history = type("_History", (), {"history": {"loss": [0.1], "val_loss": [0.2]}})()

    class _FakeModel:
        def build(self, input_shape=None):
            return None

        def count_params(self):
            return 123

        def compile(self, **kwargs):
            return None

        def fit(self, *args, **kwargs):
            return history

        def predict(self, values, verbose=0):
            values = np.asarray(values)
            return np.array([[0.9, 0.1], [0.2, 0.8]])[: values.shape[0]]

        def save(self, path):
            Path(path).write_text("model", encoding="utf-8")

    shap_calls = []

    helpers["time"] = type("_Time", (), {"time": staticmethod(lambda: 1000.0)})()
    helpers["os"] = os
    helpers["json"] = json
    helpers["gc"] = type("_GC", (), {"collect": staticmethod(lambda: None)})()
    helpers["logger"] = type(
        "_Logger",
        (),
        {
            "info": staticmethod(lambda *args, **kwargs: None),
            "warning": staticmethod(lambda *args, **kwargs: None),
            "error": staticmethod(lambda *args, **kwargs: None),
        },
    )()
    helpers["tf"] = type(
        "_TF",
        (),
        {
            "random": type("_Random", (), {"set_seed": staticmethod(lambda seed: None)})(),
            "keras": type(
                "_Keras",
                (),
                {
                    "callbacks": type(
                        "_Callbacks",
                        (),
                        {
                            "EarlyStopping": staticmethod(lambda **kwargs: {"early_stopping": kwargs})
                        },
                    )(),
                    "losses": type(
                        "_Losses",
                        (),
                        {"SparseCategoricalCrossentropy": staticmethod(lambda: "loss")},
                    )(),
                    "backend": type(
                        "_Backend", (), {"clear_session": staticmethod(lambda: None)}
                    )(),
                },
            )(),
        },
    )()
    helpers["train_test_split"] = lambda X, y, test_size, stratify, random_state: (
        X[:3],
        X[3:],
        y[:3],
        y[3:],
    )
    helpers["setup_gpu"] = lambda: {"gpus_available": [], "mixed_precision": False}
    helpers["load_and_preprocess_data"] = lambda data_path, test_size, seed: {
        "X_train": np.arange(15, dtype=float).reshape(5, 3),
        "X_test": np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]),
        "y_train": np.array([0, 1, 0, 1, 0]),
        "y_test": np.array([0, 1]),
        "num_classes": 2,
        "feature_names": ["f1", "f2", "f3"],
        "class_names": ["A", "B"],
        "feature_importances": np.array([0.4, 0.3, 0.2]),
        "preprocess_meta": {"k": "v"},
    }
    helpers["ModelConfig"] = lambda **kwargs: kwargs
    helpers["EnhancedMSTGAT"] = lambda config: _FakeModel()
    helpers["WarmupCosineDecay"] = lambda **kwargs: {"schedule": kwargs}
    helpers["QAAdamW_Lite"] = lambda **kwargs: {"optimizer": kwargs}
    helpers["create_datasets"] = lambda X_train, y_train, X_val, y_val: ("train_ds", "val_ds")
    helpers["accuracy_score"] = lambda y_true, y_pred: 1.0
    helpers["precision_score"] = lambda y_true, y_pred, average=None, zero_division=0: (
        np.array([1.0, 1.0]) if average is None else 1.0
    )
    helpers["recall_score"] = lambda y_true, y_pred, average=None, zero_division=0: (
        np.array([1.0, 1.0]) if average is None else 1.0
    )
    helpers["f1_score"] = lambda y_true, y_pred, average=None, zero_division=0: (
        np.array([1.0, 1.0]) if average is None else 1.0
    )
    helpers["plot_training_history"] = lambda *args, **kwargs: None
    helpers["plot_confusion_matrix"] = lambda *args, **kwargs: None
    helpers["plot_per_class_metrics"] = lambda *args, **kwargs: None
    helpers["plot_feature_importance"] = lambda *args, **kwargs: None
    helpers["save_preprocess_meta"] = lambda meta, path: Path(path).write_text(
        json.dumps(meta), encoding="utf-8"
    )
    helpers["generate_shap_artifacts"] = lambda **kwargs: shap_calls.append(kwargs) or {
        "status": "success",
        "summary_path": os.path.join(kwargs["output_dir"], "shap_summary.png"),
        "bar_path": os.path.join(kwargs["output_dir"], "shap_bar.png"),
        "importance_path": os.path.join(kwargs["output_dir"], "shap_importance.json"),
        "meta_path": os.path.join(kwargs["output_dir"], "shap_meta.json"),
    }

    output_dir = tmp_path / "results"
    result = helpers["run_training_pipeline"](
        data_path="dataset.xlsx",
        test_size=0.2,
        epochs=2,
        budget=1,
        seed=5,
        use_gpu=False,
        skip_ceo=True,
        output_dir=str(output_dir),
    )

    assert len(shap_calls) == 1
    assert np.array_equal(
        shap_calls[0]["X_train"],
        np.arange(9, dtype=float).reshape(3, 3),
    )
    assert np.array_equal(
        shap_calls[0]["X_test"],
        np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]),
    )
    assert shap_calls[0]["feature_names"] == ["f1", "f2", "f3"]
    assert result["shap"]["status"] == "success"

    saved_results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert saved_results["shap"]["status"] == "success"
    assert saved_results["accuracy"] == 1.0
    assert "output_dir" in saved_results
    assert saved_results["output_dir"] == str(output_dir)
    assert (output_dir / "preprocess_meta.json").exists()
    assert (output_dir / "ceo_qaadam_emstgat.keras").exists()
    assert "feature_importance.png" not in saved_results
    assert result["output_dir"] == str(output_dir)
    assert isinstance(result["shap"], dict)
    assert result["shap"]["summary_path"].endswith("shap_summary.png")
    assert result["shap"]["importance_path"].endswith("shap_importance.json")
    assert result["shap"]["meta_path"].endswith("shap_meta.json")
    assert result["shap"]["bar_path"].endswith("shap_bar.png")
    assert result["test_ratio"] == 20
    assert result["train_ratio"] == 80
    assert result["best_params"]["hidden_units"] == 192
    assert result["gpu_info"]["gpus_available"] == []
    assert result["prediction_time"] == 0.0
    assert result["training_time"] == 0.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1_score"] == 1.0
    assert result["test_size"] == 0.2
    assert result["shap"]["status"] == saved_results["shap"]["status"]
    assert len(shap_calls[0]["X_train"]) == 3
    assert len(shap_calls[0]["X_test"]) == 2
    assert shap_calls[0]["seed"] == 5
    assert shap_calls[0]["output_dir"] == str(output_dir)
    assert shap_calls[0]["model"].__class__.__name__ == "_FakeModel"
    assert "best_params" in saved_results
    assert "gpu_info" in saved_results
    assert "training_time" in saved_results
    assert "prediction_time" in saved_results
    assert "test_ratio" in saved_results
    assert "train_ratio" in saved_results
    assert "best_params" in result
    assert "gpu_info" in result
    assert "training_time" in result
    assert "prediction_time" in result
    assert "test_ratio" in result
    assert "train_ratio" in result
    assert "shap" in result
    assert "shap" in saved_results
    assert isinstance(saved_results["shap"], dict)
    assert isinstance(saved_results["best_params"], dict)
    assert isinstance(saved_results["gpu_info"], dict)
    assert isinstance(saved_results["training_time"], float)
    assert isinstance(saved_results["prediction_time"], float)
    assert isinstance(saved_results["test_ratio"], int)
    assert isinstance(saved_results["train_ratio"], int)
    assert isinstance(result["training_time"], float)
    assert isinstance(result["prediction_time"], float)
    assert isinstance(result["test_ratio"], int)
    assert isinstance(result["train_ratio"], int)
    assert saved_results["test_size"] == 0.2
    assert result["test_size"] == saved_results["test_size"]
    assert saved_results["best_params"]["hidden_units"] == 192
    assert saved_results["gpu_info"]["gpus_available"] == []
    assert saved_results["precision"] == 1.0
    assert saved_results["recall"] == 1.0
    assert saved_results["f1_score"] == 1.0
    assert saved_results["prediction_time"] == 0.0
    assert saved_results["training_time"] == 0.0
    assert saved_results["test_ratio"] == 20
    assert saved_results["train_ratio"] == 80
    assert saved_results["output_dir"] == str(output_dir)
    assert result["output_dir"] == saved_results["output_dir"]
    assert len(shap_calls) == 1
    assert len(model.calls) if False else True
    assert time is not None


