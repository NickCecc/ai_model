from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import numpy as np
import tensorflow as tf
import keras
import joblib
import sklearn
from pathlib import Path
import json
import subprocess
import sys
from typing import Any

try:
    from api.ml.data import create_sequences, load_feature_frame, split_scale_sequences
    from api.ml.constants import CLIMATE_COLUMNS, FEATURE_COLUMNS, WEATHER_COLUMNS
    from api.ml.pipeline import PipelineOptions, prepare_model_window
    from api.ml.continuous import run_incremental_update
    from api.ml.mpc import (
        DEFAULT_CONTROL_VARIABLES,
        DEFAULT_CONTROL_WEIGHTS,
        DEFAULT_TARGET_SETPOINTS,
        DEFAULT_TARGET_WEIGHTS,
        WEATHER_VARIABLES,
        run_mpc_feedback_loop,
    )
except ModuleNotFoundError:
    from ml.data import create_sequences, load_feature_frame, split_scale_sequences
    from ml.constants import CLIMATE_COLUMNS, FEATURE_COLUMNS, WEATHER_COLUMNS
    from ml.pipeline import PipelineOptions, prepare_model_window
    from ml.continuous import run_incremental_update
    from ml.mpc import (
        DEFAULT_CONTROL_VARIABLES,
        DEFAULT_CONTROL_WEIGHTS,
        DEFAULT_TARGET_SETPOINTS,
        DEFAULT_TARGET_WEIGHTS,
        WEATHER_VARIABLES,
        run_mpc_feedback_loop,
    )

app = FastAPI()

# Resolve paths relative to this file so the app works no matter the CWD.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
checkpoint_dir = BASE_DIR / "checkpoint"
ui_path = BASE_DIR / "ui.html"
sample_request_path = BASE_DIR / "example_request1.json"
metadata_path = checkpoint_dir / "model_metadata.json"

# Load the model
model = keras.models.load_model(checkpoint_dir / "model.keras")
# Load the scalers
feature_scaler = joblib.load(checkpoint_dir / "feature_scaler.save")
target_scaler = joblib.load(checkpoint_dir / "target_scaler.save")

if metadata_path.exists():
    model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
else:
    model_metadata = {
        "architecture": "legacy_lstm",
        "lookback": 20,
        "features": FEATURE_COLUMNS,
    }

EXPECTED_LOOKBACK = int(model_metadata.get("lookback", 20))
EXPECTED_FEATURES = list(model_metadata.get("features", FEATURE_COLUMNS))
MODEL_INPUT_KIND = str(model_metadata.get("architecture_kind", "single_input"))
CLIMATE_INDICES = [EXPECTED_FEATURES.index(c) for c in CLIMATE_COLUMNS if c in EXPECTED_FEATURES]
WEATHER_INDICES = [EXPECTED_FEATURES.index(c) for c in WEATHER_COLUMNS if c in EXPECTED_FEATURES]
TARGET_FEATURES = list(model_metadata.get("target_features", ["Tair", "CO2air", "HumDef"]))


class InputData(BaseModel):
    data: list[list[float]] | None = None
    use_automated_pipeline: bool = False
    automated_pipeline: dict[str, Any] = Field(default_factory=dict)


class RunScriptRequest(BaseModel):
    path: str
    args: list[str] = []
    timeout_seconds: int = 25


class MPCSimulationRequest(BaseModel):
    data: list[list[float]] | None = None
    use_automated_pipeline: bool = False
    automated_pipeline: dict[str, Any] = Field(default_factory=dict)
    steps: int = Field(default=12, ge=1, le=240)
    horizon: int = Field(default=6, ge=1, le=48)
    candidate_sequences: int = Field(default=80, ge=1, le=4000)
    random_seed: int = 42
    target_setpoints: dict[str, float] = Field(default_factory=dict)
    target_weights: dict[str, float] = Field(default_factory=dict)
    control_weights: dict[str, float] = Field(default_factory=dict)
    control_bounds: dict[str, list[float]] = Field(default_factory=dict)
    weather_forecast: list[dict[str, float]] = Field(default_factory=list)


class ExplainRequest(BaseModel):
    data: list[list[float]] | None = None
    use_automated_pipeline: bool = False
    automated_pipeline: dict[str, Any] = Field(default_factory=dict)
    method: str = "gradient_attention"
    target_feature: str | None = None
    top_k_features: int = Field(default=8, ge=1, le=21)


class MPCScenario(BaseModel):
    name: str
    target_setpoints: dict[str, float] = Field(default_factory=dict)
    target_weights: dict[str, float] = Field(default_factory=dict)
    control_weights: dict[str, float] = Field(default_factory=dict)
    control_bounds: dict[str, list[float]] = Field(default_factory=dict)
    weather_forecast: list[dict[str, float]] = Field(default_factory=list)


class MPCScenarioEvaluationRequest(BaseModel):
    data: list[list[float]] | None = None
    use_automated_pipeline: bool = False
    automated_pipeline: dict[str, Any] = Field(default_factory=dict)
    steps: int = Field(default=12, ge=1, le=240)
    horizon: int = Field(default=6, ge=1, le=48)
    candidate_sequences: int = Field(default=80, ge=1, le=4000)
    random_seed: int = 42
    scenarios: list[MPCScenario]


class ContinuousLearningRequest(BaseModel):
    climate_csv: str | None = None
    weather_csv: str | None = None
    new_rows_limit: int = Field(default=4000, ge=200, le=50000)
    fine_tune_epochs: int = Field(default=2, ge=1, le=20)
    batch_size: int = Field(default=64, ge=8, le=512)
    dry_run: bool = False


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return ui_path.read_text(encoding="utf-8")


@app.get("/sample")
def sample_request():
    if not sample_request_path.exists():
        raise HTTPException(status_code=404, detail="Sample request not found")
    return json.loads(sample_request_path.read_text())


@app.get("/model-info")
def model_info():
    return {
        "architecture": model_metadata.get("architecture", "unknown"),
        "architecture_kind": MODEL_INPUT_KIND,
        "lookback": EXPECTED_LOOKBACK,
        "horizon": model_metadata.get("horizon"),
        "feature_count": len(EXPECTED_FEATURES),
        "features": EXPECTED_FEATURES,
        "climate_features": model_metadata.get("climate_features"),
        "weather_features": model_metadata.get("weather_features"),
        "metrics": model_metadata.get("metrics"),
    }


@app.get("/dataset-window")
def dataset_window(split: str = "test", index: int = 0):
    climate_csv = PROJECT_ROOT / "greenhouse_code" / "GreenhouseClimate.csv"
    weather_csv = PROJECT_ROOT / "greenhouse_code" / "Weather.csv"

    if not climate_csv.exists() or not weather_csv.exists():
        raise HTTPException(status_code=404, detail="Climate or weather CSV file not found")

    frame = load_feature_frame(climate_csv=climate_csv, weather_csv=weather_csv)
    x_values, y_values = create_sequences(
        frame=frame,
        lookback=EXPECTED_LOOKBACK,
        horizon=int(model_metadata.get("horizon", 1)),
    )

    splits = split_scale_sequences(
        x_values=x_values,
        y_values=y_values,
        train_ratio=0.7,
        val_ratio=0.15,
    )

    split = split.lower().strip()
    if split == "train":
        x_source = splits.x_train
    elif split == "val":
        x_source = splits.x_val
    elif split == "test":
        x_source = splits.x_test
    else:
        raise HTTPException(status_code=400, detail="split must be one of: train, val, test")

    if len(x_source) == 0:
        raise HTTPException(status_code=404, detail=f"No rows available for split '{split}'")

    if index < 0 or index >= len(x_source):
        raise HTTPException(status_code=400, detail=f"index out of range for split '{split}'")

    # convert scaled window back to raw/original values for dashboard use
    raw_window = feature_scaler.inverse_transform(x_source[index])

    merged_rows = []
    for t, row in enumerate(raw_window):
        row_dict = {"%time": f"{split}_{index}_{t}"}
        for i, feature_name in enumerate(EXPECTED_FEATURES):
            row_dict[feature_name] = float(row[i])
        merged_rows.append(row_dict)

    return {
        "split": split,
        "index": index,
        "lookback": EXPECTED_LOOKBACK,
        "feature_count": len(EXPECTED_FEATURES),
        "merged_rows": merged_rows,
        "data": raw_window.tolist(),
    }


@app.get("/model/comparison")
def model_comparison():
    summary_path = checkpoint_dir / "training_summary.json"
    comparison_path = checkpoint_dir / "model_comparison.json"

    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    if comparison_path.exists():
        return json.loads(comparison_path.read_text(encoding="utf-8"))

    raise HTTPException(status_code=404, detail="No model comparison found. Run train_hybrid_models.py first.")


@app.get("/pipeline/default-config")
def pipeline_default_config():
    return {
        "lookback": EXPECTED_LOOKBACK,
        "merge_strategy": "inner",
        "fill_method": "interpolate",
        "scaling_mode": "trained",
        "supported_merge_strategies": ["inner", "left", "right", "outer"],
        "supported_fill_methods": ["interpolate", "ffill", "bfill", "zero"],
        "supported_scaling_modes": ["trained", "dynamic"],
        "feature_schema": EXPECTED_FEATURES,
    }


@app.get("/explain/default-config")
def explain_default_config():
    default_target = "Tair" if "Tair" in EXPECTED_FEATURES else EXPECTED_FEATURES[0]
    return {
        "methods": ["gradient_attention", "shap_approx"],
        "default_method": "gradient_attention",
        "default_target_feature": default_target,
        "top_k_features": 8,
        "lookback": EXPECTED_LOOKBACK,
        "features": EXPECTED_FEATURES,
    }


@app.get("/mpc/default-config")
def mpc_default_config():
    return {
        "lookback": EXPECTED_LOOKBACK,
        "feature_count": len(EXPECTED_FEATURES),
        "control_variables": [var for var in DEFAULT_CONTROL_VARIABLES if var in EXPECTED_FEATURES],
        "weather_variables": [var for var in WEATHER_VARIABLES if var in EXPECTED_FEATURES],
        "target_setpoints": DEFAULT_TARGET_SETPOINTS,
        "target_weights": DEFAULT_TARGET_WEIGHTS,
        "control_weights": DEFAULT_CONTROL_WEIGHTS,
        "recommended": {
            "steps": 12,
            "horizon": 6,
            "candidate_sequences": 80,
        },
        "scenario_templates": [
            {"name": "Balanced", "target_setpoints": {"Tair": 21.0, "CO2air": 500.0, "HumDef": 6.0}},
            {"name": "EnergySaver", "target_setpoints": {"Tair": 20.0, "CO2air": 470.0}, "control_weights": {"_magnitude": 0.08}},
            {"name": "HighGrowth", "target_setpoints": {"Tair": 22.5, "CO2air": 650.0}, "control_weights": {"_magnitude": 0.01}},
        ],
    }


def _prepare_window_from_payload(payload: dict[str, Any]):
    lookback = int(payload.get("lookback", EXPECTED_LOOKBACK))
    options = PipelineOptions(
        lookback=lookback,
        merge_strategy=str(payload.get("merge_strategy", "inner")),
        fill_method=str(payload.get("fill_method", "interpolate")),
        scaling_mode=str(payload.get("scaling_mode", "trained")),
    )

    greenhouse_rows = payload.get("greenhouse_rows", []) or []
    weather_rows = payload.get("weather_rows", []) or []
    merged_rows = payload.get("merged_rows", []) or []

    if not isinstance(greenhouse_rows, list) or not isinstance(weather_rows, list) or not isinstance(merged_rows, list):
        raise ValueError("greenhouse_rows, weather_rows, and merged_rows must be arrays")

    prepared = prepare_model_window(
        greenhouse_rows=greenhouse_rows,
        weather_rows=weather_rows,
        merged_rows=merged_rows,
        options=options,
        trained_feature_scaler=feature_scaler,
    )
    return prepared


def _resolve_input_window(
    *,
    data: list[list[float]] | None,
    use_automated_pipeline: bool,
    automated_pipeline: dict[str, Any],
):
    expected_feature_count = len(EXPECTED_FEATURES)
    if use_automated_pipeline:
        prepared = _prepare_window_from_payload(automated_pipeline)
        raw_window = np.asarray(prepared.raw_window, dtype=np.float64)
        scaled_window = np.asarray(prepared.scaled_window, dtype=np.float64)
        scaler_used = prepared.scaler_used
        pipeline_metadata = prepared.metadata
    else:
        if data is None:
            raise ValueError("data is required when use_automated_pipeline is false")
        raw_window = np.array(data, dtype=np.float64)
        expected_shape = (EXPECTED_LOOKBACK, expected_feature_count)
        if raw_window.shape != expected_shape:
            raise ValueError(f"Expected input shape {expected_shape}")
        scaled_window = feature_scaler.transform(raw_window)
        scaler_used = feature_scaler
        pipeline_metadata = None

    if not np.isfinite(raw_window).all() or not np.isfinite(scaled_window).all():
        raise ValueError("Input contains non-finite values after preprocessing")

    return raw_window, scaled_window, scaler_used, pipeline_metadata


def _target_feature_index(target_feature: str | None) -> tuple[int, str]:
    if target_feature and target_feature in EXPECTED_FEATURES:
        idx = EXPECTED_FEATURES.index(target_feature)
        return idx, target_feature
    fallback = "Tair" if "Tair" in EXPECTED_FEATURES else EXPECTED_FEATURES[0]
    return EXPECTED_FEATURES.index(fallback), fallback


def _build_model_input_from_scaled(scaled_window: np.ndarray):
    window = np.asarray(scaled_window, dtype=np.float32)
    if MODEL_INPUT_KIND == "multi_input":
        if not CLIMATE_INDICES or not WEATHER_INDICES:
            raise ValueError("Model metadata requires multi-input but climate/weather feature split is unavailable")
        climate = window[:, CLIMATE_INDICES].reshape(1, window.shape[0], len(CLIMATE_INDICES))
        weather = window[:, WEATHER_INDICES].reshape(1, window.shape[0], len(WEATHER_INDICES))
        return [climate, weather]
    return window.reshape(1, window.shape[0], window.shape[1])


def _predict_scaled_window(scaled_window: np.ndarray) -> np.ndarray:
    model_input = _build_model_input_from_scaled(scaled_window)
    return model.predict(model_input, verbose=0)


@app.post("/pipeline/prepare")
def pipeline_prepare(request: dict[str, Any]):
    try:
        prepared = _prepare_window_from_payload(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "metadata": prepared.metadata,
        "window_shape": list(prepared.raw_window.shape),
        "raw_window": prepared.raw_window.tolist(),
        "scaled_window": prepared.scaled_window.tolist(),
    }


@app.get("/files")
def list_files():
    files = []
    for p in PROJECT_ROOT.rglob("*.py"):
        if ".venv" in p.parts or "__pycache__" in p.parts:
            continue
        try:
            rel = p.relative_to(PROJECT_ROOT)
        except ValueError:
            continue
        files.append(str(rel))
    files.sort()
    return {"files": files}


@app.get("/file")
def get_file(path: str):
    file_path = (PROJECT_ROOT / path).resolve()
    if not str(file_path).startswith(str(PROJECT_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if file_path.suffix != ".py":
        raise HTTPException(status_code=400, detail="Only .py files are supported")
    return {
        "path": str(file_path.relative_to(PROJECT_ROOT)),
        "content": file_path.read_text(encoding="utf-8"),
    }


@app.post("/run-script")
def run_script(req: RunScriptRequest):
    script_path = (PROJECT_ROOT / req.path).resolve()
    if not str(script_path).startswith(str(PROJECT_ROOT.resolve())):
        raise HTTPException(status_code=400, detail="Invalid script path")
    if not script_path.exists() or not script_path.is_file():
        raise HTTPException(status_code=404, detail="Script not found")
    if script_path.suffix != ".py":
        raise HTTPException(status_code=400, detail="Only .py scripts are supported")
    if ".venv" in script_path.parts:
        raise HTTPException(status_code=400, detail="Cannot execute scripts in .venv")
    if req.timeout_seconds < 1 or req.timeout_seconds > 120:
        raise HTTPException(status_code=400, detail="timeout_seconds must be between 1 and 120")

    cmd = [sys.executable, str(script_path), *req.args]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=req.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "timed_out": True,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + f"\nTimed out after {req.timeout_seconds}s",
            "command": cmd,
        }

    return {
        "ok": result.returncode == 0,
        "timed_out": False,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": cmd,
    }


@app.post("/predict")
def predict(input: InputData):
    try:
        _raw_window, scaled_window, _scaler_used, pipeline_metadata = _resolve_input_window(
            data=input.data,
            use_automated_pipeline=input.use_automated_pipeline,
            automated_pipeline=input.automated_pipeline,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    prediction_scaled = _predict_scaled_window(scaled_window)
    prediction_original = target_scaler.inverse_transform(prediction_scaled)

    target_names = TARGET_FEATURES
    pred_values = prediction_original[0].tolist()

    return {
        "prediction": {
            target_names[i]: pred_values[i] for i in range(min(len(target_names), len(pred_values)))
        },
        "used_automated_pipeline": bool(input.use_automated_pipeline),
        "pipeline_metadata": pipeline_metadata,
    }


@app.post("/mpc/simulate")
def mpc_simulate(input: MPCSimulationRequest):
    try:
        input_array, _scaled_window, scaler_for_mpc, pipeline_metadata = _resolve_input_window(
            data=input.data,
            use_automated_pipeline=input.use_automated_pipeline,
            automated_pipeline=input.automated_pipeline,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = run_mpc_feedback_loop(
            initial_history=input_array,
            feature_names=EXPECTED_FEATURES,
            target_feature_names=TARGET_FEATURES,
            model=model,
            feature_scaler=scaler_for_mpc,
            target_scaler=target_scaler,
            steps=input.steps,
            horizon=input.horizon,
            candidate_sequences=input.candidate_sequences,
            random_seed=input.random_seed,
            target_setpoints=input.target_setpoints,
            target_weights=input.target_weights,
            control_weights=input.control_weights,
            control_bounds=input.control_bounds,
            weather_forecast=input.weather_forecast,
            predict_fn=_predict_scaled_window,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result["used_automated_pipeline"] = bool(input.use_automated_pipeline)
    result["pipeline_metadata"] = pipeline_metadata
    return result


@app.post("/mpc/evaluate-scenarios")
def mpc_evaluate_scenarios(input: MPCScenarioEvaluationRequest):
    if not input.scenarios:
        raise HTTPException(status_code=400, detail="At least one scenario is required")

    try:
        input_array, _scaled_window, scaler_for_mpc, pipeline_metadata = _resolve_input_window(
            data=input.data,
            use_automated_pipeline=input.use_automated_pipeline,
            automated_pipeline=input.automated_pipeline,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    evaluations = []
    for scenario in input.scenarios:
        try:
            scenario_result = run_mpc_feedback_loop(
                initial_history=input_array.copy(),
                feature_names=EXPECTED_FEATURES,
                target_feature_names=TARGET_FEATURES,
                model=model,
                feature_scaler=scaler_for_mpc,
                target_scaler=target_scaler,
                steps=input.steps,
                horizon=input.horizon,
                candidate_sequences=input.candidate_sequences,
                random_seed=input.random_seed,
                target_setpoints=scenario.target_setpoints,
                target_weights=scenario.target_weights,
                control_weights=scenario.control_weights,
                control_bounds=scenario.control_bounds,
                weather_forecast=scenario.weather_forecast,
                predict_fn=_predict_scaled_window,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Scenario '{scenario.name}' failed: {exc}") from exc

        evaluations.append(
            {
                "name": scenario.name,
                "energy_efficiency_score": scenario_result.get("energy_efficiency_score"),
                "stability_index": scenario_result.get("stability_index"),
                "objective_final": scenario_result.get("objective_trace", [None])[-1],
                "result": scenario_result,
            }
        )

    ranked = sorted(
        evaluations,
        key=lambda x: (
            -(x["energy_efficiency_score"] if x["energy_efficiency_score"] is not None else -1.0),
            x["stability_index"] if x["stability_index"] is not None else float("inf"),
        ),
    )

    return {
        "used_automated_pipeline": bool(input.use_automated_pipeline),
        "pipeline_metadata": pipeline_metadata,
        "scenarios_evaluated": len(evaluations),
        "ranking": [
            {
                "rank": i + 1,
                "name": item["name"],
                "energy_efficiency_score": item["energy_efficiency_score"],
                "stability_index": item["stability_index"],
                "objective_final": item["objective_final"],
            }
            for i, item in enumerate(ranked)
        ],
        "evaluations": evaluations,
    }


@app.post("/continuous/update")
def continuous_update(request: ContinuousLearningRequest):
    climate_csv = Path(request.climate_csv) if request.climate_csv else (PROJECT_ROOT / "greenhouse_code" / "GreenhouseClimate.csv")
    weather_csv = Path(request.weather_csv) if request.weather_csv else (PROJECT_ROOT / "greenhouse_code" / "Weather.csv")

    if not climate_csv.exists() or not weather_csv.exists():
        raise HTTPException(status_code=404, detail="Climate or weather CSV file not found")

    try:
        result = run_incremental_update(
            checkpoint_dir=checkpoint_dir,
            model=model,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            climate_csv=climate_csv,
            weather_csv=weather_csv,
            lookback=EXPECTED_LOOKBACK,
            architecture_kind=MODEL_INPUT_KIND,
            fine_tune_epochs=request.fine_tune_epochs,
            batch_size=request.batch_size,
            new_rows_limit=request.new_rows_limit,
            dry_run=request.dry_run,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result


@app.post("/explain")
def explain(input: ExplainRequest):
    try:
        raw_window, scaled_window, scaler_used, pipeline_metadata = _resolve_input_window(
            data=input.data,
            use_automated_pipeline=input.use_automated_pipeline,
            automated_pipeline=input.automated_pipeline,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    resolved_target_feature = input.target_feature if input.target_feature in TARGET_FEATURES else (TARGET_FEATURES[0] if TARGET_FEATURES else "Tair")
    if resolved_target_feature not in TARGET_FEATURES:
        raise HTTPException(status_code=400, detail="Target feature is not one of the trained prediction targets")

    target_idx = TARGET_FEATURES.index(resolved_target_feature)
    method = input.method.strip().lower()
    lookback, feature_count = scaled_window.shape

    base_pred_scaled = _predict_scaled_window(scaled_window)
    base_pred = target_scaler.inverse_transform(base_pred_scaled)[0]
    base_target_value = float(base_pred[target_idx])

    heatmap = np.zeros((lookback, feature_count), dtype=np.float64)

    if method == "gradient_attention":
        x_full = tf.convert_to_tensor(scaled_window.reshape(1, lookback, feature_count), dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_full)
            if MODEL_INPUT_KIND == "multi_input":
                x_climate = tf.gather(x_full, CLIMATE_INDICES, axis=2)
                x_weather = tf.gather(x_full, WEATHER_INDICES, axis=2)
                pred = model([x_climate, x_weather], training=False)
            else:
                pred = model(x_full, training=False)
            target = pred[0, target_idx]
        grads = tape.gradient(target, x_full)
        if grads is None:
            raise HTTPException(status_code=500, detail="Could not compute gradients for explanation")
        grad_values = grads.numpy()[0].astype(np.float64)
        heatmap = np.abs(grad_values * scaled_window)

    elif method == "shap_approx":
        baseline_row_raw = np.mean(raw_window, axis=0, dtype=np.float64)
        baseline_window_raw = np.tile(baseline_row_raw, (lookback, 1))

        for t in range(lookback):
            for f in range(feature_count):
                perturbed_raw = raw_window.copy()
                perturbed_raw[t, f] = baseline_window_raw[t, f]
                perturbed_scaled = scaler_used.transform(perturbed_raw)
                pred_scaled = _predict_scaled_window(perturbed_scaled)
                pred_raw = target_scaler.inverse_transform(pred_scaled)[0]
                heatmap[t, f] = base_target_value - float(pred_raw[target_idx])
        heatmap = np.abs(heatmap)

    else:
        raise HTTPException(
            status_code=400,
            detail="method must be 'gradient_attention' or 'shap_approx'",
        )

    if not np.isfinite(heatmap).all():
        raise HTTPException(status_code=500, detail="Explanation heatmap contains non-finite values")

    feature_scores = heatmap.mean(axis=0)
    timestep_scores = heatmap.mean(axis=1)
    top_k = min(input.top_k_features, len(EXPECTED_FEATURES))
    top_feature_indices = np.argsort(feature_scores)[::-1][:top_k]

    top_features = [
        {
            "feature": EXPECTED_FEATURES[int(idx)],
            "score": float(feature_scores[int(idx)]),
            "rank": rank + 1,
        }
        for rank, idx in enumerate(top_feature_indices)
    ]

    trend_summary = []
    for idx, feature in enumerate(EXPECTED_FEATURES):
        series = raw_window[:, idx]
        trend_summary.append(
            {
                "feature": feature,
                "latest": float(series[-1]),
                "mean": float(np.mean(series)),
                "min": float(np.min(series)),
                "max": float(np.max(series)),
            }
        )

    return {
        "method": method,
        "target_feature": resolved_target_feature,
        "target_prediction": base_target_value,
        "target_features": TARGET_FEATURES,
        "features": EXPECTED_FEATURES,
        "raw_window": raw_window.tolist(),
        "heatmap": heatmap.tolist(),
        "feature_importance": [
            {"feature": EXPECTED_FEATURES[i], "score": float(feature_scores[i])}
            for i in range(len(EXPECTED_FEATURES))
        ],
        "timestep_importance": [float(x) for x in timestep_scores.tolist()],
        "top_features": top_features,
        "trend_summary": trend_summary,
        "used_automated_pipeline": bool(input.use_automated_pipeline),
        "pipeline_metadata": pipeline_metadata,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)