from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from graph import AuditState, build_compiled_graph


NODE_KEYS = {
    "research": ["company_context"],
    "validation": ["retry_count"],
    "ideation": ["proposed_use_cases"],
    "reporting": ["final_report"],
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = build_compiled_graph()
    yield


app = FastAPI(
    title="LangGraph Audit API",
    description="API d'audit agentique basée sur LangGraph + Gemini.",
    version="1.0.0",
    lifespan=lifespan,
)


class AuditRequest(BaseModel):
    company_name: str = Field(..., min_length=2)


@app.post("/api/v1/audit", summary="Lance un audit", tags=["audit"])
async def audit(request: AuditRequest, req: Request):
    initial_state: AuditState = {
        "company_name": request.company_name,
        "company_context": "",
        "proposed_use_cases": [],
        "final_report": "",
        "retry_count": 0,
    }

    async def event_stream():
        try:
            async for event in req.app.state.graph.astream_events(initial_state, version="v1"):
                event_type = event.get("event")
                if event_type != "on_chain_end":
                    continue

                node_name = event.get("name")
                output = event.get("data", {}).get("output", {})

                keys = NODE_KEYS.get(node_name, [])
                filtered_output = {k: output.get(k) for k in keys if k in output}

                if not filtered_output and node_name not in NODE_KEYS:
                    continue

                payload = {"node": node_name, "output": filtered_output}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        except ValidationError as exc:
            error_payload = {"error": f"Invalid report format: {exc}"}
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        except Exception as exc:
            error_payload = {"error": str(exc)}
            yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health", summary="Health check", tags=["ops"])
def health():
    return {"status": "ok"}