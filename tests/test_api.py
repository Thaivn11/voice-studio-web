from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_home_page_serves_web_ui():
    response = client.get("/")
    assert response.status_code == 200
    assert "Voice Studio" in response.text


def test_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    assert "state" in response.json()


def test_generation_requires_model():
    import app.api as api
    old_engine = api._engine
    try:
        api._engine = None
        response = client.post("/api/generate", data={"text": "Hello", "mode": "auto"})
        assert response.status_code == 409
    finally:
        api._engine = old_engine


def test_clone_requires_explicit_consent():
    import app.api as api
    old_engine = api._engine
    try:
        api._engine = object()
        response = client.post("/api/generate", data={"text": "Hello", "mode": "clone"})
        assert response.status_code == 400
        assert "permission" in response.json()["detail"]
    finally:
        api._engine = old_engine


def test_audio_endpoint_rejects_traversal():
    response = client.get("/api/audio/..%2Fhistory.json")
    assert response.status_code in {404, 422}
