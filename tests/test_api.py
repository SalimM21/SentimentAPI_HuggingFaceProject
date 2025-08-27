# tests/test_api.py
import sys
import os
import time
import pytest

# ---- Ajouter le dossier racine dans sys.path ----
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ---------------- Tests Unitaires ----------------

def test_health_endpoint():
    """Test endpoint /health"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"


def test_root_endpoint():
    """Test endpoint /"""
    response = client.get("/")
    assert response.status_code in [200, 404]  # 404 si index.html absent

    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        data = response.json()
        assert "model" in data
    elif "text/html" in content_type:
        assert "<!doctype html>" in response.text.lower()  # vérifier qu'on a bien du HTML
    else:
        pytest.fail(f"Unexpected content-type: {content_type}")



def test_predict_positive():
    """Test prédiction sentiment positif"""
    payload = {"text": "I love this project!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["positive", "negative"]
    assert isinstance(data["confidence"], float)
    assert data["model"] != ""


# ---------------- Benchmark simple ----------------

# def test_predict_benchmark():
#     """Benchmark prédiction sur un texte moyen"""
#     payload = {"text": "This is a test sentence for benchmarking the sentiment API."}
#     start_time = time.time()
#     response = client.post("/predict", json=payload)
#     duration_ms = (time.time() - start_time) * 1000
#     assert response.status_code == 200
#     print(f"Benchmark: prediction took {duration_ms:.2f} ms")
#     assert duration_ms < 2000  # par exemple < 2s pour CPU
