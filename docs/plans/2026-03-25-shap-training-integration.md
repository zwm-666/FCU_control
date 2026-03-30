# SHAP Training Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add SHAP-based global model explanation to the training pipeline so each training run emits SHAP figures and machine-readable SHAP importance data.

**Architecture:** Keep the current training and evaluation flow unchanged, then append a post-training SHAP analysis stage inside `model/model.py`. Implement SHAP as an optional enhancement: if SHAP is unavailable or fails, preserve the existing training outputs and record the failure reason instead of aborting the whole run.

**Tech Stack:** Python, TensorFlow/Keras, NumPy, Matplotlib, SHAP, pytest

---

### Task 1: Add focused tests for SHAP aggregation and metadata export

**Files:**
- Modify: `model/test_feature_policy.py`
- Reference: `model/model.py`

**Step 1: Write the failing tests**

Add tests for:
- aggregating multi-class SHAP arrays into one global importance vector
- exporting sorted SHAP importance JSON rows with `feature` and `importance`
- exporting SHAP metadata with sample counts and status fields

Example skeleton:

```python
def test_aggregate_multiclass_shap_values_returns_global_importance():
    shap_values = [
        np.array([[1.0, -2.0], [3.0, -4.0]]),
        np.array([[0.5, -1.0], [1.5, -2.0]]),
    ]

    result = model_module.aggregate_shap_global_importance(shap_values)

    np.testing.assert_allclose(result, np.array([1.5, 2.25]))
```

```python
def test_build_shap_importance_records_sorts_descending():
    rows = model_module.build_shap_importance_records(
        ["f1", "f2", "f3"], np.array([0.2, 0.8, 0.4])
    )

    assert [row["feature"] for row in rows] == ["f2", "f3", "f1"]
```

**Step 2: Run the targeted tests and confirm they fail**

Run:
`python -m pytest "C:/Users/86191/Desktop/h2-fcu-modern-dashboard/model/test_feature_policy.py" -k shap -q`

Expected: FAIL because the SHAP helper functions do not exist yet.

**Step 3: Implement minimal helper functions**

Add helper functions to `model/model.py` for:
- aggregating SHAP values
- building sorted JSON-serializable importance rows
- building SHAP metadata payloads

Keep these pure and easy to unit test.

**Step 4: Re-run the targeted tests**

Run:
`python -m pytest "C:/Users/86191/Desktop/h2-fcu-modern-dashboard/model/test_feature_policy.py" -k shap -q`

Expected: PASS.

---

### Task 2: Add optional SHAP generation utilities to the training pipeline

**Files:**
- Modify: `model/model.py`
- Reference: `model/results_testdata_80_20/`

**Step 1: Write failing tests for graceful fallback behavior**

Add tests covering:
- when SHAP import fails, the helper returns a structured failure payload instead of raising
- when SHAP analysis succeeds, output paths and JSON payloads are produced

Use monkeypatching to simulate:
- `import shap` failure
- a fake explainer returning deterministic SHAP values
- no actual training needed

**Step 2: Run the targeted tests and confirm they fail**

Run:
`python -m pytest "C:/Users/86191/Desktop/h2-fcu-modern-dashboard/model/test_feature_policy.py" -k "shap and fallback" -q`

Expected: FAIL because no SHAP pipeline helper exists yet.

**Step 3: Implement SHAP generation helpers in `model/model.py`**

Add:
- optional SHAP import inside helper scope
- background sample selection from `X_train`
- explanation sample selection from `X_test`
- `predict_fn` wrapper using `model.predict(..., verbose=0)`
- global importance aggregation across classes
- figure generation for:
  - `shap_summary.png`
  - `shap_bar.png`
- JSON generation for:
  - `shap_importance.json`
  - `shap_meta.json`

Implementation constraints:
- fix random seed usage for reproducibility
- keep sample sizes modest and configurable via constants
- catch SHAP-specific exceptions and return `{status: "failed" | "skipped"}` payloads
- do not remove existing `feature_importance.png`

**Step 4: Re-run the targeted tests**

Run:
`python -m pytest "C:/Users/86191/Desktop/h2-fcu-modern-dashboard/model/test_feature_policy.py" -k shap -q`

Expected: PASS.

---

### Task 3: Wire SHAP outputs into `run_training_pipeline`

**Files:**
- Modify: `model/model.py:1830-1885`
- Test: `model/test_feature_policy.py`

**Step 1: Write a failing integration-style test**

Add a test that monkeypatches the training pipeline’s SHAP helper and asserts:
- SHAP helper is called with processed train/test arrays and feature names
- returned SHAP summary is inserted into `results.json` payload
- existing outputs remain untouched

Use monkeypatching to avoid real TensorFlow training.

**Step 2: Run the focused test and confirm it fails**

Run:
`python -m pytest "C:/Users/86191/Desktop/h2-fcu-modern-dashboard/model/test_feature_policy.py" -k "training_pipeline and shap" -q`

Expected: FAIL because training pipeline does not yet include SHAP outputs.

**Step 3: Implement the wiring**

Update `run_training_pipeline` to:
- call the new SHAP helper after existing plots are generated
- store returned SHAP status/details under a `shap` key in `results`
- continue saving `results.json` even if SHAP status is `failed` or `skipped`

Example shape:

```json
{
  "accuracy": 0.99,
  "f1_score": 0.99,
  "shap": {
    "status": "success",
    "summary_path": "results_testdata_80_20/shap_summary.png",
    "bar_path": "results_testdata_80_20/shap_bar.png",
    "importance_path": "results_testdata_80_20/shap_importance.json",
    "meta_path": "results_testdata_80_20/shap_meta.json"
  }
}
```

**Step 4: Re-run the focused test**

Run:
`python -m pytest "C:/Users/86191/Desktop/h2-fcu-modern-dashboard/model/test_feature_policy.py" -k "training_pipeline and shap" -q`

Expected: PASS.

---

### Task 4: Add dependency declaration for SHAP

**Files:**
- Modify: `requirements` location if present; otherwise modify the project’s Python dependency documentation/source-of-truth used by model execution
- Likely modify: `README.md` only if it already documents Python model dependencies
- Verify actual dependency source before editing

**Step 1: Identify the Python dependency source of truth**

Check whether the project uses:
- `requirements.txt`
- `pyproject.toml`
- README-based manual setup only

**Step 2: Write the minimal failing verification**

If dependency source exists, verify SHAP is not declared yet.

**Step 3: Add `shap` minimally**

Only update the actual dependency source of truth. Do not invent extra packaging structure.

**Step 4: Verify the declaration exists**

Run the appropriate verification command, e.g.:
- `python - <<'PY' ...` to inspect dependency file contents

Expected: SHAP dependency is declared exactly once.

---

### Task 5: Run the full relevant test suite

**Files:**
- Test: `model/test_feature_policy.py`
- Test: `model/test_preprocess_utils.py`

**Step 1: Run the model unit tests**

Run:
`python -m pytest "C:/Users/86191/Desktop/h2-fcu-modern-dashboard/model/test_feature_policy.py" "C:/Users/86191/Desktop/h2-fcu-modern-dashboard/model/test_preprocess_utils.py" -q`

Expected: PASS.

**Step 2: If tests fail, fix one root cause at a time**

Do not batch speculative fixes.

**Step 3: Re-run the full relevant tests**

Run the same pytest command again.

Expected: PASS with zero failures.

---

### Task 6: Run one fresh end-to-end training verification

**Files:**
- Modify if needed: `model/model.py`
- Output: a temporary or existing `results_*` directory under `model/`

**Step 1: Run the training entry point with a small, realistic verification configuration**

Use the existing training command or entry point already used by this repository.
If runtime is too high, run the smallest supported configuration that still executes the SHAP stage.

**Step 2: Verify outputs exist**

Confirm that the run produces:
- `shap_summary.png`
- `shap_bar.png`
- `shap_importance.json`
- `shap_meta.json`
- updated `results.json` containing the `shap` section

**Step 3: Verify failure semantics if SHAP cannot run**

If the environment blocks SHAP execution, confirm training still completes and `results.json` records a non-success SHAP status rather than crashing.

---

### Task 7: Update the Word report generator to prefer SHAP figures when present

**Files:**
- Modify: `.claude/generate_report_with_figures.py`

**Step 1: Write the minimal failing logic test or manual verification target**

Define the desired precedence:
- if `shap_summary.png` / `shap_bar.png` exist, use them in the report
- otherwise fall back to the current `feature_importance.png`

**Step 2: Implement the minimal change**

Update the report generator so the representative results section can include SHAP outputs without breaking existing report generation.

**Step 3: Regenerate the DOCX and verify media count increases or image selection changes as expected**

Run the report generator and inspect the generated DOCX zip contents.

Expected: DOCX still builds successfully and includes the intended explanatory figures.

---

### Task 8: Final verification

**Files:**
- Verify: `model/model.py`
- Verify: `model/test_feature_policy.py`
- Verify: `.claude/generate_report_with_figures.py`

**Step 1: Run all final verification commands fresh**

Run:
`python -m pytest "C:/Users/86191/Desktop/h2-fcu-modern-dashboard/model/test_feature_policy.py" "C:/Users/86191/Desktop/h2-fcu-modern-dashboard/model/test_preprocess_utils.py" -q`

Then run the report generation verification command.

**Step 2: Inspect outputs, not just exit codes**

Confirm:
- tests are green
- SHAP files are present or gracefully skipped
- report generation still succeeds

**Step 3: Only then report completion**

State exactly what was verified and what artifacts were produced.
