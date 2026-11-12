from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_root_page_works():
    response = client.get('/')
    assert response.status_code == 200
    assert 'Voice Studio' in response.text


def test_generation_requires_model_loaded():
    response = client.post('/api/generate', data={'text': 'Hello world', 'mode': 'auto'})
    assert response.status_code == 409
