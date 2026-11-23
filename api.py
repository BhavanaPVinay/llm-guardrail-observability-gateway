"""FastAPI application for LLM Guardrail & Observability Gateway."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from gateway import GatewayResponse, GatewayService
from observability import ObservabilityStore, default_store


class ChatRequest(BaseModel):
    """Payload for POST /chat."""

    prompt: str = Field(..., description="The user prompt to be evaluated and processed")
    session_id: Optional[str] = Field(default=None, description="Optional session or conversation ID")


class HealthResponse(BaseModel):
    """Payload for GET /health."""

    status: str = "healthy"
    service: str = "llm-guardrail-observability-gateway"


def create_app(
    gateway: Optional[GatewayService] = None,
    store: Optional[ObservabilityStore] = None,
) -> FastAPI:
    """Create and configure the FastAPI gateway application."""
    obs_store = store or default_store
    gw_service = gateway or GatewayService(store=obs_store)

    application = FastAPI(
        title="LLM Guardrail & Observability Gateway",
        description="Deterministic guardrail validation, PII redaction, and telemetry for LLMs",
        version="1.0.0",
    )

    # Attach to application state for easy test/lifecycle access
    application.state.gateway = gw_service
    application.state.store = obs_store

    @application.get("/health", response_model=HealthResponse, tags=["System"])
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse()

    @application.post("/chat", response_model=GatewayResponse, tags=["Gateway"])
    async def chat(request: ChatRequest) -> GatewayResponse:
        """Evaluate input guardrails, redact PII, invoke responder, and evaluate output guardrails."""
        return await gw_service.aprocess(prompt=request.prompt, session_id=request.session_id)

    @application.get("/metrics", response_model=Dict[str, Any], tags=["Observability"])
    async def metrics() -> Dict[str, Any]:
        """Retrieve aggregated metrics: total_requests, allowed_requests, blocked_requests, redacted_requests, avg_latency_ms."""
        return obs_store.get_metrics()

    @application.get("/traces", response_model=List[Dict[str, Any]], tags=["Observability"])
    async def traces(
        limit: int = Query(default=100, ge=1, le=1000, description="Max number of traces to return")
    ) -> List[Dict[str, Any]]:
        """Retrieve recent request traces."""
        return obs_store.get_traces(limit=limit)

    return application


# Standard application instance for uvicorn run: `uvicorn api:app`
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
