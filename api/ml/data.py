from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .constants import CLIMATE_COLUMNS, FEATURE_COLUMNS, WEATHER_COLUMNS, TARGET_COLUMNS


@dataclass
class DatasetSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler


def load_feature_frame(climate_csv: Path, weather_csv: Path) -> pd.DataFrame:
    climate = pd.read_csv(climate_csv, low_memory=False)
    weather = pd.read_csv(weather_csv, low_memory=False)

    required_climate = {"%time", *CLIMATE_COLUMNS}
    required_weather = {"%time", *WEATHER_COLUMNS}
    missing_climate = sorted(required_climate.difference(climate.columns))
    missing_weather = sorted(required_weather.difference(weather.columns))
    if missing_climate:
        raise ValueError(f"Climate CSV is missing columns: {missing_climate}")
    if missing_weather:
        raise ValueError(f"Weather CSV is missing columns: {missing_weather}")

    climate = climate[["%time", *CLIMATE_COLUMNS]].copy()
    weather = weather[["%time", *WEATHER_COLUMNS]].copy()

    merged = climate.merge(weather, on="%time", how="inner")
    merged = merged.sort_values("%time").drop_duplicates(subset=["%time"], keep="last")

    frame = merged[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    all_nan_cols = [col for col in FEATURE_COLUMNS if frame[col].isna().all()]
    if all_nan_cols:
        raise ValueError(f"Columns are entirely non-numeric/missing after coercion: {all_nan_cols}")

    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.interpolate(method="linear", limit_direction="both", axis=0)
    frame = frame.ffill().bfill()
    frame = frame.fillna(frame.median(numeric_only=True))
    frame = frame.astype(np.float32)
    if frame.empty:
        raise ValueError("No usable rows after cleaning/interpolation")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Cleaned feature frame still contains non-finite values")

    return frame


def create_sequences(frame: pd.DataFrame, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    data = frame.to_numpy(dtype=np.float32)
    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []

    max_start = len(data) - lookback - horizon + 1
    if max_start <= 0:
        raise ValueError("Dataset is too small for requested lookback/horizon")

    for start in range(max_start):
        end = start + lookback
        target_idx = end + horizon - 1
        x_values.append(data[start:end, :])
        target_indices = [FEATURE_COLUMNS.index(col) for col in TARGET_COLUMNS]
        y_values.append(data[target_idx, target_indices])

    return np.stack(x_values), np.stack(y_values)


def split_scale_sequences(
    x_values: np.ndarray,
    y_values: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> DatasetSplits:
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must be in (0,1) and sum to < 1")

    total = len(x_values)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    x_train, x_val, x_test = x_values[:train_end], x_values[train_end:val_end], x_values[val_end:]
    y_train, y_val, y_test = y_values[:train_end], y_values[train_end:val_end], y_values[val_end:]

    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise ValueError("Split produced an empty partition; adjust ratios or dataset size")

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    x_train_scaled = feature_scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_val_scaled = feature_scaler.transform(x_val.reshape(-1, x_val.shape[-1])).reshape(x_val.shape)
    x_test_scaled = feature_scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

    y_train_scaled = target_scaler.fit_transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)
    y_test_scaled = target_scaler.transform(y_test)

    arrays = {
        "x_train_scaled": x_train_scaled,
        "x_val_scaled": x_val_scaled,
        "x_test_scaled": x_test_scaled,
        "y_train_scaled": y_train_scaled,
        "y_val_scaled": y_val_scaled,
        "y_test_scaled": y_test_scaled,
    }
    bad = [name for name, arr in arrays.items() if not np.isfinite(arr).all()]
    if bad:
        raise ValueError(f"Non-finite values detected after scaling: {bad}")

    return DatasetSplits(
        x_train=x_train_scaled,
        y_train=y_train_scaled,
        x_val=x_val_scaled,
        y_val=y_val_scaled,
        x_test=x_test_scaled,
        y_test=y_test_scaled,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )
