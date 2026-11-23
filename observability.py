"""Observability store and telemetry tracking for LLM Gateway."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class RequestTrace(BaseModel):
    """Detailed telemetry record for an individual request/response lifecycle."""

    trace_id: str
    timestamp: float = Field(default_factory=time.time)
    session_id: Optional[str] = None
    prompt: str
    response: Optional[str] = None
    status: str  # "allowed", "blocked", "error"
    allowed: bool
    latency_ms: float
    flags: List[str] = Field(default_factory=list)
    redactions: List[str] = Field(default_factory=list)
    reason: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricsSummary(BaseModel):
    """Aggregated gateway metrics."""

    total_requests: int = 0
    allowed_requests: int = 0
    blocked_requests: int = 0
    redacted_requests: int = 0
    avg_latency_ms: float = 0.0


class ObservabilityStore:
    """In-memory telemetry and metrics store with thread-safe recording and reset."""

    def __init__(self, max_traces: int = 1000) -> None:
        self.max_traces = max_traces
        self._lock = threading.Lock()
        self._traces: List[RequestTrace] = []
        self._total_requests: int = 0
        self._allowed_requests: int = 0
        self._blocked_requests: int = 0
        self._redacted_requests: int = 0
        self._total_latency_ms: float = 0.0

    def record(self, trace_or_data: Union[RequestTrace, Dict[str, Any]]) -> RequestTrace:
        """Record a request/response event and update aggregated metrics."""
        if isinstance(trace_or_data, dict):
            trace = RequestTrace(**trace_or_data)
        else:
            trace = trace_or_data

        with self._lock:
            self._total_requests += 1
            if trace.allowed:
                self._allowed_requests += 1
            else:
                self._blocked_requests += 1

            if len(trace.redactions) > 0:
                self._redacted_requests += 1

            self._total_latency_ms += trace.latency_ms

            self._traces.append(trace)
            if len(self._traces) > self.max_traces:
                self._traces.pop(0)

        return trace

    def record_event(self, trace_or_data: Union[RequestTrace, Dict[str, Any]]) -> RequestTrace:
        """Alias for record()."""
        return self.record(trace_or_data)

    def get_metrics(self) -> Dict[str, Union[int, float]]:
        """Compute and return aggregated metrics summary as a dictionary."""
        with self._lock:
            avg_lat = (
                round(self._total_latency_ms / self._total_requests, 2)
                if self._total_requests > 0
                else 0.0
            )
            return {
                "total_requests": self._total_requests,
                "allowed_requests": self._allowed_requests,
                "blocked_requests": self._blocked_requests,
                "redacted_requests": self._redacted_requests,
                "avg_latency_ms": avg_lat,
            }

    def get_metrics_summary(self) -> MetricsSummary:
        """Return metrics as a structured MetricsSummary model."""
        return MetricsSummary(**self.get_metrics())

    def get_traces(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve recent traces, optionally capped by limit (most recent first)."""
        with self._lock:
            # Return shallow copy of trace dicts, newest first
            traces_slice = list(reversed(self._traces))
            if limit is not None and limit > 0:
                traces_slice = traces_slice[:limit]
            return [trace.model_dump() for trace in traces_slice]

    def reset(self) -> None:
        """Reset all counters, traces, and metrics (useful for clean test runs)."""
        with self._lock:
            self._traces.clear()
            self._total_requests = 0
            self._allowed_requests = 0
            self._blocked_requests = 0
            self._redacted_requests = 0
            self._total_latency_ms = 0.0


# Shared default instance
default_store = ObservabilityStore()
