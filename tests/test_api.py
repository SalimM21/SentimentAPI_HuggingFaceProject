from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model" in data

def test_predict_success():
    r = client.post("/predict", json={"text": "I love this!"})
    assert r.status_code == 200
    data = r.json()
    assert data["label"] in ("positive", "negative")
    assert 0.0 <= data["confidence"] <= 1.0
    assert "model" in data

def test_predict_error_empty():
    r = client.post("/predict", json={"text": ""})
    # Pydantic validation error -> 422
    assert r.status_code == 422
    body = r.json()
    assert "text must be a non-empty string" in str(body)