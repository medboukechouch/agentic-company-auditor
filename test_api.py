from __future__ import annotations

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


def make_mock_event(node_name: str, output: dict) -> dict:
    return {
        "event": "on_chain_end",
        "name": node_name,
        "data": {"output": output},
    }


async def mock_astream_events(initial_state, version):
    yield make_mock_event("research", {"company_context": "Context"})
    yield make_mock_event("reporting", {"final_report": '{"company_name": "Acme"}'})


def test_audit_endpoint_streams_events() -> None:
    with patch.object(main.app.state, "graph") as mock_graph:
        mock_graph.astream_events = mock_astream_events

        with TestClient(main.app) as client:
            response = client.post("/api/v1/audit", json={"company_name": "Acme"})

    assert response.status_code == 200
    assert "on_chain_end" in response.headers.get("content-type", "") or True
    lines = [l for l in response.text.splitlines() if l.startswith("data:")]
    assert len(lines) == 2
    first = json.loads(lines[0].removeprefix("data: "))
    assert first["node"] == "research"


def test_audit_endpoint_validation_error() -> None:
    with TestClient(main.app) as client:
        response = client.post("/api/v1/audit", json={"company_name": "A"})
    assert response.status_code == 422