import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    with patch("src.uphora.src.uphora.backend.agent.agent.ChatDatabricks"), \
         patch("src.uphora.src.uphora.backend.agent.agent.mlflow"):
        from src.uphora.src.uphora.backend.app import app
        return TestClient(app)

def test_list_customers_returns_10(client):
    with patch("src.uphora.src.uphora.backend.router._fetch_demo_customers") as mock_fetch:
        mock_fetch.return_value = [
            {"id": f"cust_{i:05d}", "name": f"Customer {i}", "skin_type": "oily", "skin_tone": "medium"}
            for i in range(1, 11)
        ]
        resp = client.get("/api/customers")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 10

def test_chat_returns_sse_stream(client, sample_products):
    from mlflow.types.responses import ResponsesAgentStreamEvent

    text_item = MagicMock()
    text_item.text = "Hello there!"
    fake_event = MagicMock()
    fake_event.type = "response.output_item.done"
    fake_event.item = text_item

    with patch("src.uphora.src.uphora.backend.router.AGENT") as mock_agent, \
         patch("src.uphora.src.uphora.backend.router.save_memory_sync"):

        mock_agent.predict_stream.return_value = iter([fake_event])
        mock_agent.get_last_products.return_value = sample_products[:2]

        resp = client.post("/api/chat", json={
            "customer_id": "cust_00001",
            "message": "What serum for oily skin?",
            "history": [],
        })

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
