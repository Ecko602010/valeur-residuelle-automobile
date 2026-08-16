
import json
from pathlib import Path

import pytest

from app import create_app


BASE_DIR = Path(__file__).resolve().parents[1]
SAMPLE_PATH = BASE_DIR / "tests" / "sample_payload.json"


@pytest.fixture()
def client():
    application = create_app({
        "TESTING": True,
    })

    with application.test_client() as test_client:
        yield test_client


def load_sample():
    return json.loads(
        SAMPLE_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200

    payload = response.get_json()

    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


def test_homepage(client):
    response = client.get("/")

    assert response.status_code == 200

    page = response.get_data(
        as_text=True
    )

    assert "Valeur résiduelle" in page
    assert 'id="load-demo"' in page


def test_html_prediction(client):
    response = client.post(
        "/estimer",
        data=load_sample(),
    )

    assert response.status_code == 200

    page = response.get_data(
        as_text=True
    )

    assert (
        "Valeur résiduelle estimée"
        in page
    )

    assert (
        "Prix de marché actuel prédit"
        in page
    )


def test_api_prediction(client):
    response = client.post(
        "/api/v1/predict",
        json=load_sample(),
    )

    assert response.status_code == 200

    payload = response.get_json()
    result = payload["result"]

    assert (
        result[
            "valeur_residuelle_estimee_eur"
        ]
        > 0
    )

    assert (
        5
        <= result[
            "valeur_residuelle_estimee_pct"
        ]
        <= 100
    )


def test_prediction_intervals_include_central_value(
    client,
):
    response = client.post(
        "/api/v1/predict",
        json=load_sample(),
    )

    assert response.status_code == 200

    result = response.get_json()[
        "result"
    ]

    if (
        "prix_marche_actuel_borne_basse_eur"
        in result
    ):
        assert (
            result[
                "prix_marche_actuel_borne_basse_eur"
            ]
            <= result[
                "prix_marche_actuel_predit_eur"
            ]
            <= result[
                "prix_marche_actuel_borne_haute_eur"
            ]
        )

    if (
        "prix_marche_futur_borne_basse_eur"
        in result
    ):
        assert (
            result[
                "prix_marche_futur_borne_basse_eur"
            ]
            <= result[
                "prix_marche_futur_brut_predit_eur"
            ]
            <= result[
                "prix_marche_futur_borne_haute_eur"
            ]
        )


def test_invalid_mileage(client):
    payload = load_sample()
    payload["mileage_km"] = -100

    response = client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_missing_required_field(client):
    payload = load_sample()
    payload.pop("make_norm")

    response = client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_security_headers(client):
    response = client.get("/")

    assert (
        response.headers[
            "X-Content-Type-Options"
        ]
        == "nosniff"
    )

    assert (
        response.headers[
            "X-Frame-Options"
        ]
        == "DENY"
    )

    assert (
        "Content-Security-Policy"
        in response.headers
    )


def test_not_found(client):
    response = client.get(
        "/page-inexistante"
    )

    assert response.status_code == 404
    assert "Page introuvable" in (
        response.get_data(
            as_text=True
        )
    )


def test_privacy_and_accessibility_pages(
    client,
):
    assert (
        client.get(
            "/confidentialite"
        ).status_code
        == 200
    )

    assert (
        client.get(
            "/accessibilite"
        ).status_code
        == 200
    )
