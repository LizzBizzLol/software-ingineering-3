from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_summarize_ok():
    payload = {"text": "Искусственный интеллект помогает людям решать сложные задачи."}
    r = client.post("/summarize", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["summary"], str) and len(data["summary"]) > 0
    assert data["tokens_in"] > 0
    assert data["tokens_out"] > 0

def test_summarize_validation_empty():
    r = client.post("/summarize", json={"text": ""})
    assert r.status_code == 400

def test_summarize_custom_params():
    r = client.post("/summarize", json={
        "text": "Модель т5 умеет суммаризировать длинные тексты.",
        "max_length": 64, "min_length": 10, "num_beams": 2
    })
    assert r.status_code == 200
