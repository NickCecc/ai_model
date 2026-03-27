# Greenhouse AI Platform

This API and dashboard provide:
- greenhouse climate prediction
- trained model comparison
- MPC simulation and scenario benchmarking
- explainability heatmaps and feature importance
- trend visualization
- incremental continuous-learning updates

## Run The App

The main web dashboard is served directly by the FastAPI app.

### 1. Create and activate a virtual environment
```bash
cd /Users/nickcecchin/Desktop/ai_model
python3 -m venv api/.venv
source api/.venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r api/requirements.txt
```

### 3. Start the API and dashboard
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Open the dashboard
Open:

`http://127.0.0.1:8000`

This serves the main UI from `api/ui.html`.

## If No Trained Checkpoint Exists

The app expects a trained checkpoint in `api/checkpoint/`.

If you do not have one yet, train the models first:

```bash
python api/train_hybrid_models.py --architecture all
```

## Model Training

Train and compare the available architectures:

```bash
python api/train_hybrid_models.py --architecture all
```

Supported architectures:
- `baseline_lstm`
- `lstm_cnn`
- `bi_lstm`
- `multi_input_hybrid`

Outputs are written to `api/checkpoint/`, including:
- per-model folders
- promoted model weights and scalers
- `model_metadata.json`
- `training_summary.json`
- `model_comparison.json`
- `model_comparison.csv`

## Benchmark Existing Checkpoints

```bash
python api/benchmark_models.py
```

This produces benchmark comparison outputs locally.

## Automated Data Processing

```bash
python api/automated_data_pipeline.py
```

This prepares merged greenhouse and weather features for modeling.

## Continuous Learning

API endpoint:
- `POST /continuous/update`

Example request:

```json
{
  "new_rows_limit": 4000,
  "fine_tune_epochs": 2,
  "batch_size": 64,
  "dry_run": false
}
```

## Core API Endpoints

### Prediction
- `POST /predict`

Supports:
- direct mode with `data`
- automated mode with `use_automated_pipeline=true`

### Pipeline
- `GET /pipeline/default-config`
- `POST /pipeline/prepare`

### MPC
- `GET /mpc/default-config`
- `POST /mpc/simulate`
- `POST /mpc/evaluate-scenarios`

### Explainability
- `GET /explain/default-config`
- `POST /explain`

### Model Metadata / Comparison
- `GET /model-info`
- `GET /model/comparison`

## Dashboard Sections

The main UI includes:
- Summary
- Predict
- Model Comparison
- MPC Control
- Explainability
- Charts / Trends
- Settings / Environment


## Git And Generated Files

Generated training outputs should stay local. The repo-level `.gitignore` is set up to avoid pushing:
- `api/checkpoint/`
- `greenhouse_code/checkpoint/`
- local virtual environments

If generated files are already tracked in Git, they must be untracked separately before `.gitignore` will stop them from appearing in commits.
