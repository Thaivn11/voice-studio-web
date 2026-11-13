from fastapi.testclient import TestClient

import app.api as api

client = TestClient(api.app)


def login(username="admin", password=None):
    if password is None:
        # Tests use a temporary administrator password configured in CI.
        password = "test-administrator-password"
    return client.post("/api/auth/login", data={"username": username, "password": password})


def test_health_and_home_are_available():
    assert client.get("/api/health").json() == {"status": "ok"}
    response = client.get("/")
    assert response.status_code == 200
    assert "Voice Studio" in response.text


def test_protected_api_requires_login():
    assert client.get("/api/overview").status_code == 401
    assert client.get("/api/projects").status_code == 401


def test_login_and_create_project(tmp_path, monkeypatch):
    from app.auth import AuthManager
    from app.project_manager import ProjectManager

    auth = AuthManager(tmp_path / "auth.json")
    auth.create_user("test-admin", "test-administrator-password", "admin")
    token = auth.issue_token("test-admin")
    assert auth.validate_token(token)["role"] == "admin"

    projects = ProjectManager(tmp_path / "projects.json")
    project = projects.create_project("test-admin", "Internal test", "test")
    assert projects.list_projects("test-admin", "admin")[0]["id"] == project["id"]


def test_generation_requires_model_and_auth():
    response = client.post("/api/generate", data={"text": "Hello", "mode": "auto"})
    assert response.status_code == 401
    old_engine = api._engine
    try:
        api._engine = None
        response = client.post(
            "/api/generate",
            data={"text": "Hello", "mode": "auto"},
            headers={"Authorization": "Bearer invalid"},
        )
        assert response.status_code == 401
    finally:
        api._engine = old_engine
