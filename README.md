# Greenhouse AI Platform

A machine learning and control platform for greenhouse climate forecasting, model benchmarking, explainability, and model predictive control.

This project combines deep learning with operational decision support. It predicts key greenhouse climate variables from recent greenhouse and weather history, compares multiple model architectures, explains model behavior, and simulates control strategies through MPC.

## Overview

The platform is built around a few core ideas:

- forecast future greenhouse conditions from recent sensor and weather data
- compare multiple neural network architectures and deploy the best-performing model
- explain predictions with feature importance and timestep heatmaps
- simulate control strategies with model predictive control
- provide a browser-based dashboard for demos, inspection, and technical use

## Main Capabilities

- **Project Summary / Dashboard**
  - high-level overview of the system, deployed model, and key outputs

- **Prediction**
  - run next-step greenhouse climate predictions from model input windows

- **Model Comparison**
  - compare trained architectures using RMSE, MAE, and related metrics

- **MPC Control / Simulation**
  - simulate control trajectories and benchmark control scenarios

- **Explainability**
  - inspect feature importance and attribution heatmaps for predictions

- **Charts / Trends**
  - visualize recent greenhouse and weather behavior over time

- **Developer Tools**
  - browse and run project Python scripts from the dashboard

- **Environment / Metadata**
  - inspect model metadata, schema details, and preprocessing defaults

## Tech Stack

- **Backend:** FastAPI
- **Frontend:** HTML, CSS, JavaScript
- **ML / Data:** TensorFlow / Keras, NumPy, scikit-learn, joblib
- **Control:** custom MPC simulation pipeline

## Repository Structure

```text
ai_model/
├── api/
│   ├── main.py
│   ├── ui.html
│   ├── train_hybrid_models.py
│   ├── benchmark_models.py
│   ├── automated_data_pipeline.py
│   ├── evaluate_mpc_scenarios.py
│   ├── ml/
│   └── requirements.txt
├── greenhouse_code/
│   ├── GreenhouseClimate.csv
│   ├── Weather.csv
│   └── ...
└── web-system/
# Greenhouse AI / ML Project

This project provides a FastAPI backend and browser dashboard for:
- greenhouse climate prediction
- model comparison
- MPC simulation
- explainability and heatmaps
- trend visualization
- developer tools and script execution

## Run The Main App

The main web dashboard is served directly by the FastAPI app. There is no separate frontend build step for the primary greenhouse UI.

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

That serves the main UI from `api/ui.html`.

## If No Model Checkpoint Exists Yet

The app expects a trained checkpoint in `api/checkpoint/`.

If you do not have one yet, train the models first:

```bash
python api/train_hybrid_models.py --architecture all
```

This generates model weights, scalers, metadata, and comparison outputs locally.

## Optional: Run The Older Static Web Folder

There is also a `web-system/` folder, but it is not the main greenhouse dashboard.

If you want to open it anyway:

```bash
cd /Users/nickcecchin/Desktop/ai_model/web-system
python3 -m http.server 8080
```

Then open:

`http://127.0.0.1:8080`

## Training Artifacts And Git

The repo-level `.gitignore` keeps generated files local by default, including:
- `api/checkpoint/`
- `greenhouse_code/checkpoint/`
- local virtual environments

Important:

If checkpoint files are already tracked in Git, `.gitignore` alone will not stop them from appearing in future commits. They must be untracked separately with `git rm --cached`.

