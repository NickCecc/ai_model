from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_CONTROL_VARIABLES = [
    "Ventwind",
    "VentLee",
    "PipeGrow",
    "AssimLight",
    "EnScr",
    "BlackScr",
    "co2_dos",
]

WEATHER_VARIABLES = [
    "Windsp",
    "Winddir",
    "Tout",
    "Rhout",
    "AbsHumOut",
    "PARout",
    "Iglob",
    "Pyrgeo",
    "RadSum",
]

DEFAULT_TARGET_SETPOINTS = {
    "CO2air": 500.0,
    "Tair": 21.0,
    "HumDef": 6.0,
}

DEFAULT_TARGET_WEIGHTS = {
    "CO2air": 1.2,
    "Tair": 5.0,
    "HumDef": 2.5,
}

# Global MPC cost weights. Per-control overrides can be passed in control_weights.
DEFAULT_CONTROL_WEIGHTS = {
    "_delta": 0.25,
    "_magnitude": 0.02,
}

PHYSICAL_BOUNDS = {
    "Ventwind": (0.0, 100.0),
    "VentLee": (0.0, 100.0),
    "PipeGrow": (0.0, 100.0),
    "AssimLight": (0.0, 100.0),
    "EnScr": (0.0, 100.0),
    "BlackScr": (0.0, 100.0),
    "co2_dos": (0.0, 100.0),
}


def _feature_index(feature_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(feature_names)}


def _state_to_dict(values: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    return {name: float(values[idx]) for idx, name in enumerate(feature_names)}


def _clip_action(action: dict[str, float], bounds: dict[str, tuple[float, float]]) -> dict[str, float]:
    clipped = {}
    for var, value in action.items():
        lo, hi = bounds[var]
        clipped[var] = float(np.clip(value, lo, hi))
    return clipped


def infer_control_bounds(
    history_window: np.ndarray,
    feature_names: list[str],
    control_variables: list[str],
) -> dict[str, tuple[float, float]]:
    index = _feature_index(feature_names)
    bounds: dict[str, tuple[float, float]] = {}

    for var in control_variables:
        series = history_window[:, index[var]].astype(float)
        finite_series = series[np.isfinite(series)]

        if finite_series.size == 0:
            low, high = 0.0, 1.0
        else:
            p10 = float(np.percentile(finite_series, 10))
            p90 = float(np.percentile(finite_series, 90))
            spread = p90 - p10
            if spread <= 1e-8:
                margin = max(abs(p10) * 0.2, 1.0)
            else:
                margin = max(spread * 0.35, 0.5)
            low = p10 - margin
            high = p90 + margin

        if var in PHYSICAL_BOUNDS:
            phy_low, phy_high = PHYSICAL_BOUNDS[var]
            low = max(low, phy_low)
            high = min(high, phy_high)

        if high <= low:
            high = low + 1.0

        bounds[var] = (float(low), float(high))

    return bounds


def _resolve_control_bounds(
    inferred_bounds: dict[str, tuple[float, float]],
    overrides: dict[str, list[float]],
) -> dict[str, tuple[float, float]]:
    bounds = inferred_bounds.copy()
    for var, override in overrides.items():
        if var not in bounds:
            continue
        if len(override) != 2:
            raise ValueError(f"Control bounds for '{var}' must be [min, max]")
        low, high = float(override[0]), float(override[1])
        if high <= low:
            raise ValueError(f"Control bounds for '{var}' are invalid: min must be < max")
        bounds[var] = (low, high)
    return bounds


def _apply_weather_override(
    row: np.ndarray,
    weather: dict[str, float],
    index: dict[str, int],
) -> None:
    for var, value in weather.items():
        if var in index:
            row[index[var]] = float(value)


def _predict_next_row(
    history_window: np.ndarray,
    action: dict[str, float],
    weather: dict[str, float],
    feature_names: list[str],
    target_feature_names: list[str],
    feature_scaler,
    target_scaler,
    model,
    predict_fn=None,
) -> np.ndarray:
    """
    Predict the next full feature row.

    The trained forecasting model may only predict a subset of features
    (e.g. Tair, CO2air, HumDef). MPC still needs a full 21-feature state to
    continue simulation, so we:
      1. start from the last known row,
      2. overwrite predicted target features,
      3. enforce commanded controls and supplied weather values.
    """
    index = _feature_index(feature_names)
    work_window = history_window.copy().astype(np.float64)

    # Apply the commanded controls to the most recent row so the model
    # conditions on the chosen actuator settings.
    for var, value in action.items():
        if var in index:
            work_window[-1, index[var]] = float(value)

    if weather:
        _apply_weather_override(work_window[-1], weather, index)

    scaled = feature_scaler.transform(work_window)

    if predict_fn is None:
        prediction_scaled = model.predict(
            scaled.reshape(1, scaled.shape[0], scaled.shape[1]),
            verbose=0,
        )
    else:
        prediction_scaled = predict_fn(scaled)

    predicted_targets = target_scaler.inverse_transform(prediction_scaled)[0].astype(np.float64)

    # Build a full next-state row starting from the previous row.
    next_row = work_window[-1].copy()

    # Overwrite only the model's predicted target features.
    for feature_name, value in zip(target_feature_names, predicted_targets):
        if feature_name in index:
            next_row[index[feature_name]] = float(value)

    # Enforce commanded controls and exogenous weather in the resulting next state.
    for var, value in action.items():
        if var in index:
            next_row[index[var]] = float(value)
    if weather:
        _apply_weather_override(next_row, weather, index)

    return np.nan_to_num(next_row, nan=0.0, posinf=1e6, neginf=-1e6)


def _compute_step_cost(
    predicted_state: np.ndarray,
    action: dict[str, float],
    previous_action: dict[str, float],
    target_setpoints: dict[str, float],
    target_weights: dict[str, float],
    control_weights: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    feature_index: dict[str, int],
) -> float:
    cost = 0.0

    for var, setpoint in target_setpoints.items():
        if var not in feature_index:
            continue
        value = float(predicted_state[feature_index[var]])
        scale = max(abs(setpoint), 1.0)
        error = (value - setpoint) / scale
        weight = float(target_weights.get(var, 1.0))
        cost += weight * (error ** 2)

    global_delta_weight = float(control_weights.get("_delta", DEFAULT_CONTROL_WEIGHTS["_delta"]))
    magnitude_weight = float(control_weights.get("_magnitude", DEFAULT_CONTROL_WEIGHTS["_magnitude"]))

    for var, value in action.items():
        low, high = bounds[var]
        span = max(high - low, 1e-6)
        delta = (value - previous_action[var]) / span
        delta_weight = float(control_weights.get(var, global_delta_weight))
        magnitude = (value - low) / span

        cost += delta_weight * (delta ** 2)
        cost += magnitude_weight * (magnitude ** 2)

    return float(cost)


def _sample_action_plan(
    rng: np.random.Generator,
    horizon: int,
    control_variables: list[str],
    bounds: dict[str, tuple[float, float]],
    previous_action: dict[str, float],
) -> list[dict[str, float]]:
    plan: list[dict[str, float]] = []
    current = previous_action.copy()

    for _ in range(horizon):
        action: dict[str, float] = {}
        for var in control_variables:
            low, high = bounds[var]
            span = high - low
            if span <= 1e-8:
                value = low
            elif rng.random() < 0.65:
                value = current[var] + rng.normal(0.0, 0.15 * span)
            else:
                value = rng.uniform(low, high)
            action[var] = float(np.clip(value, low, high))
        plan.append(action)
        current = action

    return plan


def _rollout_plan(
    history_window: np.ndarray,
    action_plan: list[dict[str, float]],
    weather_horizon: list[dict[str, float]],
    feature_names: list[str],
    target_feature_names: list[str],
    feature_scaler,
    target_scaler,
    model,
    predict_fn,
    target_setpoints: dict[str, float],
    target_weights: dict[str, float],
    control_weights: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    previous_action: dict[str, float],
) -> tuple[float, list[np.ndarray], np.ndarray]:
    feature_index = _feature_index(feature_names)
    sim_window = history_window.copy().astype(np.float64)
    total_cost = 0.0
    predictions: list[np.ndarray] = []
    last_action = previous_action.copy()

    for step_idx, action in enumerate(action_plan):
        weather = weather_horizon[step_idx] if step_idx < len(weather_horizon) else {}
        clipped_action = _clip_action(action, bounds)

        next_state = _predict_next_row(
            history_window=sim_window,
            action=clipped_action,
            weather=weather,
            feature_names=feature_names,
            target_feature_names=target_feature_names,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            model=model,
            predict_fn=predict_fn,
        )

        step_cost = _compute_step_cost(
            predicted_state=next_state,
            action=clipped_action,
            previous_action=last_action,
            target_setpoints=target_setpoints,
            target_weights=target_weights,
            control_weights=control_weights,
            bounds=bounds,
            feature_index=feature_index,
        )
        total_cost += step_cost

        predictions.append(next_state)
        sim_window = np.vstack([sim_window[1:], next_state])
        last_action = clipped_action

    return float(total_cost), predictions, sim_window


def _optimize_action_plan(
    history_window: np.ndarray,
    horizon: int,
    candidate_sequences: int,
    weather_horizon: list[dict[str, float]],
    feature_names: list[str],
    target_feature_names: list[str],
    feature_scaler,
    target_scaler,
    model,
    predict_fn,
    control_variables: list[str],
    bounds: dict[str, tuple[float, float]],
    target_setpoints: dict[str, float],
    target_weights: dict[str, float],
    control_weights: dict[str, float],
    rng: np.random.Generator,
) -> tuple[list[dict[str, float]], float, list[np.ndarray]]:
    feature_index = _feature_index(feature_names)
    previous_action = {
        var: float(history_window[-1, feature_index[var]]) for var in control_variables
    }

    hold_plan = [previous_action.copy() for _ in range(horizon)]
    best_cost, best_predictions, _ = _rollout_plan(
        history_window=history_window,
        action_plan=hold_plan,
        weather_horizon=weather_horizon,
        feature_names=feature_names,
        target_feature_names=target_feature_names,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        model=model,
        predict_fn=predict_fn,
        target_setpoints=target_setpoints,
        target_weights=target_weights,
        control_weights=control_weights,
        bounds=bounds,
        previous_action=previous_action,
    )
    best_plan = hold_plan

    for _ in range(candidate_sequences):
        candidate_plan = _sample_action_plan(
            rng=rng,
            horizon=horizon,
            control_variables=control_variables,
            bounds=bounds,
            previous_action=previous_action,
        )
        cost, predictions, _ = _rollout_plan(
            history_window=history_window,
            action_plan=candidate_plan,
            weather_horizon=weather_horizon,
            feature_names=feature_names,
            target_feature_names=target_feature_names,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            model=model,
            predict_fn=predict_fn,
            target_setpoints=target_setpoints,
            target_weights=target_weights,
            control_weights=control_weights,
            bounds=bounds,
            previous_action=previous_action,
        )
        if cost < best_cost:
            best_cost = cost
            best_plan = candidate_plan
            best_predictions = predictions

    return best_plan, float(best_cost), best_predictions


def _compute_run_metrics(
    step_results: list[dict[str, Any]],
    control_variables: list[str],
    control_bounds: dict[str, tuple[float, float]],
    tracked_features: list[str],
) -> dict[str, float]:
    if not step_results:
        return {"energy_efficiency_score": 0.0, "stability_index": 0.0}

    energy_terms = []
    for step in step_results:
        action = step["applied_action"]
        per_control = []
        for var in control_variables:
            low, high = control_bounds[var]
            span = max(high - low, 1e-9)
            per_control.append((float(action[var]) - low) / span)
        energy_terms.append(float(np.mean(per_control)))

    state_series = {f: [] for f in tracked_features}
    for step in step_results:
        state = step["predicted_state"]
        for f in tracked_features:
            if f in state:
                state_series[f].append(float(state[f]))

    stability_vals = []
    for values in state_series.values():
        if len(values) >= 2:
            stability_vals.append(float(np.std(values)))
    stability_index = float(np.mean(stability_vals)) if stability_vals else 0.0

    # Higher score means more efficient (less normalized actuator intensity).
    energy_efficiency_score = float(max(0.0, 1.0 - np.mean(energy_terms)))
    return {
        "energy_efficiency_score": energy_efficiency_score,
        "stability_index": stability_index,
    }


def run_mpc_feedback_loop(
    initial_history: np.ndarray,
    feature_names: list[str],
    target_feature_names: list[str],
    model,
    feature_scaler,
    target_scaler,
    *,
    steps: int,
    horizon: int,
    candidate_sequences: int,
    random_seed: int,
    target_setpoints: dict[str, float] | None = None,
    target_weights: dict[str, float] | None = None,
    control_weights: dict[str, float] | None = None,
    control_bounds: dict[str, list[float]] | None = None,
    weather_forecast: list[dict[str, float]] | None = None,
    predict_fn=None,
) -> dict[str, Any]:
    if steps < 1 or horizon < 1 or candidate_sequences < 1:
        raise ValueError("steps, horizon, and candidate_sequences must be >= 1")

    history = np.asarray(initial_history, dtype=np.float64)
    if not np.isfinite(history).all():
        raise ValueError("Initial history contains non-finite values")

    index = _feature_index(feature_names)
    control_variables = [var for var in DEFAULT_CONTROL_VARIABLES if var in index]
    if not control_variables:
        raise ValueError("No control variables found in the current feature schema")

    # Keep only target names that actually exist in the current full feature schema.
    resolved_target_feature_names = [name for name in target_feature_names if name in index]
    if not resolved_target_feature_names:
        raise ValueError("No target feature names match the current feature schema")

    bounds = infer_control_bounds(
        history_window=history,
        feature_names=feature_names,
        control_variables=control_variables,
    )
    bounds = _resolve_control_bounds(bounds, control_bounds or {})

    resolved_setpoints = {
        var: value
        for var, value in DEFAULT_TARGET_SETPOINTS.items()
        if var in index
    }
    if target_setpoints:
        for var, value in target_setpoints.items():
            if var in index:
                resolved_setpoints[var] = float(value)

    resolved_target_weights = DEFAULT_TARGET_WEIGHTS.copy()
    if target_weights:
        for var, value in target_weights.items():
            resolved_target_weights[var] = float(value)

    resolved_control_weights = DEFAULT_CONTROL_WEIGHTS.copy()
    if control_weights:
        for var, value in control_weights.items():
            resolved_control_weights[var] = float(value)

    weather_profile = weather_forecast or []
    rng = np.random.default_rng(random_seed)

    step_results = []
    objective_trace: list[float] = []

    for step_idx in range(steps):
        weather_horizon = weather_profile[step_idx:step_idx + horizon]
        best_plan, best_cost, best_predictions = _optimize_action_plan(
            history_window=history,
            horizon=horizon,
            candidate_sequences=candidate_sequences,
            weather_horizon=weather_horizon,
            feature_names=feature_names,
            target_feature_names=resolved_target_feature_names,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            model=model,
            predict_fn=predict_fn,
            control_variables=control_variables,
            bounds=bounds,
            target_setpoints=resolved_setpoints,
            target_weights=resolved_target_weights,
            control_weights=resolved_control_weights,
            rng=rng,
        )

        applied_action = _clip_action(best_plan[0], bounds)
        next_weather = weather_profile[step_idx] if step_idx < len(weather_profile) else {}
        next_state = _predict_next_row(
            history_window=history,
            action=applied_action,
            weather=next_weather,
            feature_names=feature_names,
            target_feature_names=resolved_target_feature_names,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            model=model,
            predict_fn=predict_fn,
        )

        history = np.vstack([history[1:], next_state])
        objective_trace.append(float(best_cost))

        step_results.append(
            {
                "step": step_idx + 1,
                "objective": float(best_cost),
                "applied_action": applied_action,
                "predicted_state": _state_to_dict(next_state, feature_names),
                "planned_actions": best_plan,
                "horizon_predictions": [_state_to_dict(row, feature_names) for row in best_predictions],
            }
        )

    run_metrics = _compute_run_metrics(
        step_results=step_results,
        control_variables=control_variables,
        control_bounds=bounds,
        tracked_features=[f for f in ("Tair", "CO2air", "HumDef") if f in feature_names],
    )

    return {
        "controller": "random-shooting-mpc",
        "steps": steps,
        "horizon": horizon,
        "candidate_sequences": candidate_sequences,
        "control_variables": control_variables,
        "target_feature_names": resolved_target_feature_names,
        "target_setpoints": resolved_setpoints,
        "target_weights": resolved_target_weights,
        "control_weights": resolved_control_weights,
        "control_bounds": {var: [float(low), float(high)] for var, (low, high) in bounds.items()},
        "objective_trace": objective_trace,
        "energy_efficiency_score": run_metrics["energy_efficiency_score"],
        "stability_index": run_metrics["stability_index"],
        "final_state": _state_to_dict(history[-1], feature_names),
        "history_window_final": history.tolist(),
        "results": step_results,
    }