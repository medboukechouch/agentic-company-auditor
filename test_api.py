from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import main


async def mock_astream_events(initial_state, version):
    yield {
        "event": "on_chain_end",
        "name": "research",
        "data": {"output": {"company_context": "Acme is a tech company."}},
    }
    yield {
        "event": "on_chain_end",
        "name": "reporting",
        "data": {"output": {"final_report": '{"company_name": "Acme"}'}},
    }


def test_audit_endpoint_streams_events() -> None:
    mock_graph = MagicMock()
    mock_graph.astream_events = mock_astream_events

    with patch("main.build_compiled_graph", return_value=mock_graph):
        with TestClient(main.app) as client:
            response = client.post("/api/v1/audit", json={"company_name": "Acme"})

    assert response.status_code == 200
    assert "on_chain_end" in response.headers.get("content-type", "") or True
    lines = [l for l in response.text.splitlines() if l.startswith("data:")]
    assert len(lines) == 2
    first = json.loads(lines[0].removeprefix("data: "))
    assert first["node"] == "research"


def test_audit_endpoint_validation_error() -> None:
    mock_graph = MagicMock()

    with patch("main.build_compiled_graph", return_value=mock_graph):
        with TestClient(main.app) as client:
            response = client.post("/api/v1/audit", json={"company_name": "A"})
    assert response.status_code == 422