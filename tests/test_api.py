from fastapi.testclient import TestClient

import api


class FakeRedisClient:
    async def ping(self):
        return True


class FakeRedis:
    def __init__(self):
        self.redis = FakeRedisClient()

    async def connect(self):
        return None

    async def close(self):
        return None


def test_health_does_not_require_llm_or_api_key(monkeypatch):
    monkeypatch.setattr(api, "Redis", FakeRedis)

    response = TestClient(api.app).get("/health")

    assert response.status_code == 200
    assert response.json()["redis"] == "ok"


def test_chat_rejects_invalid_api_key(monkeypatch):
    monkeypatch.setattr(api.settings, "API_KEY", "test-key")
    api.limiter._requests.clear()

    response = TestClient(api.app).post(
        "/api/v1/chat",
        headers={"X-API-Key": "wrong-key"},
        json={"session_id": "s", "user_id": "u", "query": "hello"},
    )

    assert response.status_code == 401


def test_chat_validates_request_size(monkeypatch):
    monkeypatch.setattr(api.settings, "API_KEY", "test-key")
    api.limiter._requests.clear()

    response = TestClient(api.app).post(
        "/api/v1/chat",
        headers={"X-API-Key": "test-key"},
        json={"session_id": "s", "user_id": "u", "query": ""},
    )

    assert response.status_code == 422