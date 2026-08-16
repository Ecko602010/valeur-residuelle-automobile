
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ENGINEERED_COLUMNS = {
    "log_mileage_km",
    "annual_mileage_estimated",
    "vehicle_age_squared",
    "vehicle_age_group",
    "mileage_group",
    "is_electric",
    "is_hybrid",
    "is_diesel",
}


def load_model_bundle(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")

    bundle = joblib.load(path)

    required_keys = {
        "model_name",
        "model",
        "feature_columns",
        "numeric_features",
        "categorical_features",
        "catboost_target_strategy",
    }

    missing = required_keys.difference(bundle)

    if missing:
        raise ValueError(
            "Bundle incomplet. Clés manquantes : "
            + ", ".join(sorted(missing))
        )

    return bundle


def _numeric_series(
    frame: pd.DataFrame,
    column: str,
    default: float = np.nan,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)

    return pd.to_numeric(frame[column], errors="coerce")


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()

    mileage = _numeric_series(enriched, "mileage_km")
    age = _numeric_series(enriched, "vehicle_age")

    enriched["log_mileage_km"] = np.log1p(
        mileage.clip(lower=0)
    )

    enriched["annual_mileage_estimated"] = (
        mileage / age.clip(lower=1)
    ).clip(lower=0, upper=100_000)

    enriched["vehicle_age_squared"] = age.pow(2)

    enriched["vehicle_age_group"] = pd.cut(
        age,
        bins=[-np.inf, 1, 3, 5, 8, 12, 20, np.inf],
        labels=[
            "0-1",
            "2-3",
            "4-5",
            "6-8",
            "9-12",
            "13-20",
            "20+",
        ],
    ).astype("object")

    enriched["mileage_group"] = pd.cut(
        mileage,
        bins=[
            -np.inf,
            20_000,
            50_000,
            100_000,
            150_000,
            200_000,
            np.inf,
        ],
        labels=[
            "0-20k",
            "20-50k",
            "50-100k",
            "100-150k",
            "150-200k",
            "200k+",
        ],
    ).astype("object")

    fuel = (
        enriched.get(
            "fuel",
            pd.Series("", index=enriched.index),
        )
        .fillna("")
        .astype(str)
        .str.lower()
    )

    primary_fuel = (
        enriched.get(
            "fuel_type_primary",
            pd.Series("", index=enriched.index),
        )
        .fillna("")
        .astype(str)
        .str.lower()
    )

    fuel_text = fuel + " " + primary_fuel

    enriched["is_electric"] = fuel_text.str.contains(
        r"electric|électrique|bev",
        regex=True,
    ).astype("int8")

    enriched["is_hybrid"] = fuel_text.str.contains(
        r"hybrid|hybride|phev",
        regex=True,
    ).astype("int8")

    enriched["is_diesel"] = fuel_text.str.contains(
        "diesel",
        regex=False,
    ).astype("int8")

    return enriched


def inverse_target(
    predictions: np.ndarray,
    strategy: str,
) -> np.ndarray:
    values = np.asarray(predictions, dtype=float)

    if strategy in {"log", "weighted_log"}:
        values = np.expm1(values)

    return np.maximum(values, 0)


def prepare_catboost_features(
    frame: pd.DataFrame,
    categorical_columns: list[str],
) -> pd.DataFrame:
    prepared = frame.copy()

    for column in categorical_columns:
        if column not in prepared.columns:
            prepared[column] = "missing"

        prepared[column] = (
            prepared[column]
            .fillna("missing")
            .astype(str)
            .replace({"": "missing"})
        )

    for column in prepared.columns:
        if column not in categorical_columns:
            prepared[column] = pd.to_numeric(
                prepared[column],
                errors="coerce",
            )

    return prepared


def prepare_application_input(
    input_frame: pd.DataFrame,
    model_bundle: dict[str, Any],
) -> pd.DataFrame:
    prepared = add_engineered_features(input_frame.copy())

    return prepared.reindex(
        columns=model_bundle["feature_columns"]
    )


def predict_market_price(
    model_bundle: dict[str, Any],
    input_frame: pd.DataFrame,
    return_interval: bool = False,
) -> dict[str, np.ndarray]:
    prepared_input = prepare_application_input(
        input_frame,
        model_bundle,
    )

    model_name = model_bundle["model_name"]
    categorical = model_bundle["categorical_features"]
    strategy = model_bundle["catboost_target_strategy"]

    if model_name == "Ensemble CatBoost + Random Forest":
        payload = model_bundle["model"]

        catboost_input = prepare_catboost_features(
            prepared_input,
            categorical,
        )

        catboost_prediction = inverse_target(
            payload["catboost_model"].predict(catboost_input),
            strategy,
        )

        random_forest_prediction = payload[
            "random_forest_model"
        ].predict(prepared_input)

        weight = float(payload["catboost_weight"])

        central_prediction = (
            weight * catboost_prediction
            + (1 - weight) * random_forest_prediction
        )

    elif model_name.startswith("CatBoost"):
        catboost_input = prepare_catboost_features(
            prepared_input,
            categorical,
        )

        central_prediction = inverse_target(
            model_bundle["model"].predict(catboost_input),
            strategy,
        )

    else:
        central_prediction = model_bundle["model"].predict(
            prepared_input
        )

    central_prediction = np.maximum(
        np.asarray(central_prediction, dtype=float),
        0,
    )

    result = {"prediction_eur": central_prediction}

    lower_model = model_bundle.get("lower_quantile_model")
    upper_model = model_bundle.get("upper_quantile_model")

    if (
        return_interval
        and lower_model is not None
        and upper_model is not None
    ):
        catboost_input = prepare_catboost_features(
            prepared_input,
            categorical,
        )

        lower = inverse_target(
            lower_model.predict(catboost_input),
            strategy,
        )

        upper = inverse_target(
            upper_model.predict(catboost_input),
            strategy,
        )

        raw_lower = np.minimum(lower, upper)
        raw_upper = np.maximum(lower, upper)





        result["borne_basse_quantile_brute_eur"] = raw_lower
        result["borne_haute_quantile_brute_eur"] = raw_upper

        result["borne_basse_eur"] = np.minimum(
            raw_lower,
            central_prediction,
        )

        result["borne_haute_eur"] = np.maximum(
            raw_upper,
            central_prediction,
        )

        result["intervalle_ajuste"] = (
            (central_prediction < raw_lower)
            | (central_prediction > raw_upper)
        )

    return result


def project_residual_value_proxy(
    model_bundle: dict[str, Any],
    vehicle_features: dict[str, Any],
    initial_acquisition_price_eur: float,
    contract_duration_months: int,
    annual_mileage_km: float,
    minimum_residual_ratio: float = 0.05,
) -> dict[str, Any]:
    if initial_acquisition_price_eur <= 0:
        raise ValueError(
            "Le prix d'acquisition doit être strictement positif."
        )

    if contract_duration_months <= 0:
        raise ValueError(
            "La durée du contrat doit être strictement positive."
        )

    if annual_mileage_km < 0:
        raise ValueError(
            "Le kilométrage annuel ne peut pas être négatif."
        )

    current_raw = pd.DataFrame([vehicle_features])

    current_result = predict_market_price(
        model_bundle,
        current_raw,
        return_interval=True,
    )

    current_market_price = float(
        current_result["prediction_eur"][0]
    )

    acquisition_to_market_ratio = (
        initial_acquisition_price_eur
        / current_market_price
        if current_market_price > 0
        else np.nan
    )

    acquisition_price_warning = bool(
        np.isfinite(
            acquisition_to_market_ratio
        )
        and (
            acquisition_to_market_ratio < 0.65
            or acquisition_to_market_ratio > 1.50
        )
    )

    projected_raw = current_raw.copy()
    duration_years = contract_duration_months / 12
    additional_mileage = annual_mileage_km * duration_years

    current_age = _numeric_series(
        projected_raw,
        "vehicle_age",
        default=0,
    ).fillna(0)

    current_mileage = _numeric_series(
        projected_raw,
        "mileage_km",
        default=0,
    ).fillna(0)

    projected_raw["vehicle_age"] = (
        current_age + duration_years
    )
    projected_raw["mileage_km"] = (
        current_mileage + additional_mileage
    )

    future_result = predict_market_price(
        model_bundle,
        projected_raw,
        return_interval=True,
    )

    future_market_price = float(
        future_result["prediction_eur"][0]
    )

    if current_market_price <= 0:
        raise RuntimeError(
            "Le prix de marché actuel prédit est nul ou négatif."
        )

    raw_ratio = future_market_price / current_market_price

    guardrail_applied = not (
        minimum_residual_ratio <= raw_ratio <= 1.0
    )

    bounded_ratio = float(
        np.clip(
            raw_ratio,
            minimum_residual_ratio,
            1.0,
        )
    )

    residual_value = (
        initial_acquisition_price_eur * bounded_ratio
    )

    output = {
        "duree_contrat_mois": int(contract_duration_months),
        "kilometrage_annuel_km": float(annual_mileage_km),
        "kilometrage_fin_contrat_km": float(
            current_mileage.iloc[0] + additional_mileage
        ),
        "prix_acquisition_initial_eur": float(
            initial_acquisition_price_eur
        ),
        "prix_marche_actuel_predit_eur": current_market_price,
        "ratio_prix_acquisition_sur_marche": float(
            acquisition_to_market_ratio
        ),
        "alerte_prix_acquisition": (
            acquisition_price_warning
        ),
        "prix_marche_futur_brut_predit_eur": future_market_price,
        "coefficient_depreciation_brut": float(raw_ratio),
        "coefficient_depreciation_retenu": bounded_ratio,
        "garde_fou_applique": bool(guardrail_applied),
        "valeur_residuelle_estimee_eur": float(residual_value),
        "valeur_residuelle_estimee_pct": float(
            bounded_ratio * 100
        ),
        "depreciation_estimee_eur": float(
            initial_acquisition_price_eur - residual_value
        ),
        "hypothese": (
            "Projection à marché constant ; "
            "proxy non contractuel"
        ),
    }

    if "borne_basse_eur" in current_result:
        output["prix_marche_actuel_borne_basse_eur"] = float(
            current_result["borne_basse_eur"][0]
        )
        output["prix_marche_actuel_borne_haute_eur"] = float(
            current_result["borne_haute_eur"][0]
        )
        output["prix_marche_actuel_intervalle_ajuste"] = bool(
            current_result.get(
                "intervalle_ajuste",
                np.array([False]),
            )[0]
        )

    if "borne_basse_eur" in future_result:
        output["prix_marche_futur_borne_basse_eur"] = float(
            future_result["borne_basse_eur"][0]
        )
        output["prix_marche_futur_borne_haute_eur"] = float(
            future_result["borne_haute_eur"][0]
        )
        output["prix_marche_futur_intervalle_ajuste"] = bool(
            future_result.get(
                "intervalle_ajuste",
                np.array([False]),
            )[0]
        )

    return output
