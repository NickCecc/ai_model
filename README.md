# Greenhouse AI Platform

A machine learning and control platform for greenhouse climate forecasting, model benchmarking, explainability, and model predictive control.

This project predicts key greenhouse climate variables from recent greenhouse and weather history, compares multiple neural network architectures, explains model behavior, and simulates control strategies through MPC.

## Project Scope

The repository combines:
- greenhouse climate prediction
- trained model comparison
- explainability and attribution views
- MPC control simulation
- a browser dashboard for demonstrations and practical inspection

## Main Features

- **Summary Dashboard**
  - overview of the project, deployed model, and key outputs

- **Prediction**
  - run next-step greenhouse climate predictions from historical input windows

- **Model Comparison**
  - compare candidate architectures using RMSE, MAE, SMAPE, and R2

- **MPC Control**
  - simulate control trajectories and benchmark control scenarios

- **Explainability**
  - inspect feature importance and timestep heatmaps

- **Charts and Trends**
  - visualize recent greenhouse and weather behavior over time

- **Settings and Environment**
  - inspect model metadata, feature schema, and preprocessing defaults

## Tech Stack

- **Backend:** FastAPI
- **Frontend:** HTML, CSS, JavaScript
- **ML / Data:** TensorFlow, Keras, NumPy, scikit-learn, joblib
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
├── docs/
├── greenhouse_code/
│   ├── GreenhouseClimate.csv
│   ├── Weather.csv
│   └── ...
└── web-system/
```

## Supported Model Architectures

- `baseline_lstm`
- `lstm_cnn`
- `bi_lstm`
- `multi_input_hybrid`
- `temporal_conv`

The platform evaluates these models on the same forecasting task and ranks them by performance metrics.

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd ai_model
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv api/.venv
source api/.venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r api/requirements.txt
```

### 4. Start the app

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Open the dashboard

Visit:

`http://127.0.0.1:8000`

## If No Trained Checkpoint Exists Yet

The app expects a trained model checkpoint in `api/checkpoint/`.

If no checkpoint exists yet, train the models first:

```bash
python api/train_hybrid_models.py --architecture all
```

## Core API Endpoints

- `POST /predict`
- `GET /model-info`
- `GET /model/comparison`
- `GET /pipeline/default-config`
- `POST /pipeline/prepare`
- `GET /mpc/default-config`
- `POST /mpc/simulate`
- `POST /mpc/evaluate-scenarios`
- `GET /explain/default-config`
- `POST /explain`
- `POST /continuous/update`

## Deliverables

The repository deliverables now include a handover report for future students:

- [Greenhouse_Project_Handover_Report.md](/Users/nickcecchin/Desktop/ai_model/docs/Greenhouse_Project_Handover_Report.md)
- [Greenhouse_Project_Handover_Report.docx](/Users/nickcecchin/Desktop/ai_model/docs/Greenhouse_Project_Handover_Report.docx)
- [Greenhouse_Project_Handover_Report.pdf](/Users/nickcecchin/Desktop/ai_model/docs/Greenhouse_Project_Handover_Report.pdf)

These documents include:
- software tools used in the project
- description of the codebase and major modules
- dataset descriptions
- installation and operation instructions
- guidance for future students continuing the work

## Training Artifacts And Git

Generated training results are intended to stay local by default.

The repo-level `.gitignore` excludes:
- `api/checkpoint/`
- `greenhouse_code/checkpoint/`
- local virtual environments

If generated files were previously tracked in Git, they need to be untracked before `.gitignore` fully prevents them from appearing in commits.

