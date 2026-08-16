
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import logging
import math
import os

import numpy as np
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
)

from model_utils import (
    load_model_bundle,
    project_residual_value_proxy,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        BASE_DIR
        / "models"
        / "modele_ml_valeur_residuelle_v2.joblib",
    )
)
OPTIONS_PATH = BASE_DIR / "data" / "form_options.json"


def load_options() -> dict[str, Any]:
    if not OPTIONS_PATH.exists():
        return {
            "brands": [],
            "models_by_brand": {},
            "countries": [],
            "vehicle_conditions": [],
            "fuels": [],
            "transmissions": [],
            "colors": [],
            "body_types": [],
            "fuel_types_primary": [],
            "vehicle_classes": [],
            "numeric_defaults": {},
        }

    return json.loads(
        OPTIONS_PATH.read_text(encoding="utf-8")
    )


def parse_optional_float(
    payload: dict[str, Any],
    key: str,
) -> float | None:
    raw_value = payload.get(key)

    if raw_value is None:
        return None

    if isinstance(raw_value, str):
        raw_value = raw_value.strip().replace(",", ".")

    if raw_value == "":
        return None

    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Le champ « {key} » doit être numérique."
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            f"Le champ « {key} » doit être un nombre fini."
        )

    return value


def parse_required_float(
    payload: dict[str, Any],
    key: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = parse_optional_float(payload, key)

    if value is None:
        raise ValueError(
            f"Le champ « {key} » est obligatoire."
        )

    if minimum is not None and value < minimum:
        raise ValueError(
            f"Le champ « {key} » doit être supérieur "
            f"ou égal à {minimum}."
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"Le champ « {key} » doit être inférieur "
            f"ou égal à {maximum}."
        )

    return value


def parse_required_int(
    payload: dict[str, Any],
    key: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = parse_required_float(
        payload,
        key,
        minimum,
        maximum,
    )

    if not float(value).is_integer():
        raise ValueError(
            f"Le champ « {key} » doit être un entier."
        )

    return int(value)


def clean_text(
    payload: dict[str, Any],
    key: str,
    required: bool = False,
    maximum_length: int = 120,
) -> str | None:
    value = payload.get(key)

    if value is None:
        value = ""

    value = str(value).strip()

    if required and not value:
        raise ValueError(
            f"Le champ « {key} » est obligatoire."
        )

    if len(value) > maximum_length:
        raise ValueError(
            f"Le champ « {key} » est trop long."
        )

    return value or None


def build_vehicle_features(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    observation_year = parse_required_int(
        payload,
        "observation_year",
        minimum=1980,
        maximum=2100,
    )

    model_year = parse_required_int(
        payload,
        "model_year",
        minimum=1950,
        maximum=observation_year + 1,
    )

    mileage_km = parse_required_float(
        payload,
        "mileage_km",
        minimum=0,
        maximum=1_000_000,
    )

    acquisition_price = parse_required_float(
        payload,
        "initial_acquisition_price_eur",
        minimum=1,
        maximum=2_000_000,
    )

    contract_duration = parse_required_int(
        payload,
        "contract_duration_months",
        minimum=6,
        maximum=96,
    )

    annual_mileage = parse_required_float(
        payload,
        "annual_mileage_km",
        minimum=0,
        maximum=100_000,
    )

    vehicle_age = max(
        observation_year - model_year,
        0,
    )

    features = {
        "make_norm": clean_text(
            payload,
            "make_norm",
            required=True,
        ),
        "model_norm": clean_text(
            payload,
            "model_norm",
            required=True,
        ),
        "country_code": clean_text(
            payload,
            "country_code",
            required=True,
            maximum_length=10,
        ),
        "vehicle_condition": clean_text(
            payload,
            "vehicle_condition",
        ),
        "fuel": clean_text(payload, "fuel"),
        "transmission": clean_text(
            payload,
            "transmission",
        ),
        "color": clean_text(payload, "color"),
        "body_type": clean_text(
            payload,
            "body_type",
        ),
        "fuel_type_primary": clean_text(
            payload,
            "fuel_type_primary",
        ),
        "vehicle_class": clean_text(
            payload,
            "vehicle_class",
        ),
        "model_year": model_year,
        "observation_year": observation_year,
        "mileage_km": mileage_km,
        "vehicle_age": vehicle_age,
        "engine_displacement_l": parse_optional_float(
            payload,
            "engine_displacement_l",
        ),
        "combined_mpg": parse_optional_float(
            payload,
            "combined_mpg",
        ),
        "co2_tailpipe_g_per_mile": parse_optional_float(
            payload,
            "co2_tailpipe_g_per_mile",
        ),
        "electric_range_miles": parse_optional_float(
            payload,
            "electric_range_miles",
        ),
    }

    contract = {
        "initial_acquisition_price_eur": acquisition_price,
        "contract_duration_months": contract_duration,
        "annual_mileage_km": annual_mileage,
    }

    return features, contract


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_for_json(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(item) for item in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def create_app(
    test_config: dict[str, Any] | None = None,
) -> Flask:
    app = Flask(__name__)

    app.config.from_mapping(
        MAX_CONTENT_LENGTH=32 * 1024,
        JSON_SORT_KEYS=False,
    )

    if test_config:
        app.config.update(test_config)

    app.logger.setLevel(logging.INFO)

    model_bundle = load_model_bundle(MODEL_PATH)
    form_options = load_options()

    @app.after_request
    def add_security_headers(response):
        response.headers[
            "X-Content-Type-Options"
        ] = "nosniff"
        response.headers[
            "X-Frame-Options"
        ] = "DENY"
        response.headers[
            "Referrer-Policy"
        ] = "strict-origin-when-cross-origin"
        response.headers[
            "Permissions-Policy"
        ] = (
            "camera=(), microphone=(), "
            "geolocation=()"
        )
        response.headers[
            "Content-Security-Policy"
        ] = (
            "default-src 'self'; "
            "style-src 'self'; "
            "script-src 'self'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "form-action 'self'; "
            "frame-ancestors 'none';"
        )
        return response

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            options=form_options,
            form_data={},
            error=None,
        )

    @app.post("/estimer")
    def estimate():
        payload = request.form.to_dict()

        try:
            features, contract = build_vehicle_features(
                payload
            )

            result = project_residual_value_proxy(
                model_bundle=model_bundle,
                vehicle_features=features,
                initial_acquisition_price_eur=contract[
                    "initial_acquisition_price_eur"
                ],
                contract_duration_months=contract[
                    "contract_duration_months"
                ],
                annual_mileage_km=contract[
                    "annual_mileage_km"
                ],
            )

            last_training_year = int(
                form_options.get(
                    "dataset_observation_year_max",
                    features["observation_year"],
                )
            )

            result["derniere_annee_donnees"] = (
                last_training_year
            )
            result["extrapolation_temporelle"] = bool(
                features["observation_year"]
                > last_training_year
            )

            return render_template(
                "resultat.html",
                result=result,
                vehicle=features,
                model_name=model_bundle["model_name"],
                metrics=model_bundle.get("metrics", {}),
            )

        except ValueError as exc:
            return (
                render_template(
                    "index.html",
                    options=form_options,
                    form_data=payload,
                    error=str(exc),
                ),
                400,
            )

        except Exception:
            app.logger.exception(
                "Erreur de prédiction."
            )
            return (
                render_template(
                    "erreur.html",
                    message=(
                        "La prédiction n'a pas pu être "
                        "calculée. Vérifiez les données "
                        "saisies ou réessayez."
                    ),
                ),
                500,
            )

    @app.post("/api/v1/predict")
    def predict_api():
        payload = request.get_json(
            silent=True
        )

        if not isinstance(payload, dict):
            return jsonify({
                "error": (
                    "Le corps de la requête doit être "
                    "un objet JSON."
                )
            }), 400

        try:
            features, contract = build_vehicle_features(
                payload
            )

            result = project_residual_value_proxy(
                model_bundle=model_bundle,
                vehicle_features=features,
                initial_acquisition_price_eur=contract[
                    "initial_acquisition_price_eur"
                ],
                contract_duration_months=contract[
                    "contract_duration_months"
                ],
                annual_mileage_km=contract[
                    "annual_mileage_km"
                ],
            )

            last_training_year = int(
                form_options.get(
                    "dataset_observation_year_max",
                    features["observation_year"],
                )
            )

            result["derniere_annee_donnees"] = (
                last_training_year
            )
            result["extrapolation_temporelle"] = bool(
                features["observation_year"]
                > last_training_year
            )

            return jsonify(
                _sanitize_for_json({
                    "model": model_bundle["model_name"],
                    "result": result,
                })
            )

        except ValueError as exc:
            return jsonify({
                "error": str(exc)
            }), 400

        except Exception:
            app.logger.exception(
                "Erreur API de prédiction."
            )
            return jsonify({
                "error": (
                    "Erreur interne lors de la prédiction."
                )
            }), 500

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "model_name": model_bundle["model_name"],
        })

    @app.get("/confidentialite")
    def privacy():
        return render_template(
            "confidentialite.html"
        )

    @app.get("/accessibilite")
    def accessibility():
        return render_template(
            "accessibilite.html"
        )

    @app.errorhandler(404)
    def not_found(_error):
        return (
            render_template(
                "erreur.html",
                message="Page introuvable.",
            ),
            404,
        )

    @app.errorhandler(413)
    def too_large(_error):
        return (
            render_template(
                "erreur.html",
                message=(
                    "La requête envoyée est trop volumineuse."
                ),
            ),
            413,
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
