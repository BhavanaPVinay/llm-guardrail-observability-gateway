"""LLM Guardrail & Observability Gateway Service."""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from typing import Any, Callable, List, Optional
from pydantic import BaseModel, Field

from guardrails import check_input_guardrails, check_output_guardrails
from observability import ObservabilityStore, RequestTrace, default_store


class GatewayResponse(BaseModel):
    """Structured response returned by GatewayService."""

    request_id: str
    allowed: bool
    status: str  # "allowed", "blocked", or "error"
    output: Optional[str] = None
    redacted_prompt: str
    flags: List[str] = Field(default_factory=list)
    redactions: List[str] = Field(default_factory=list)
    reason: str
    latency_ms: float
    session_id: Optional[str] = None


def default_responder(prompt: str, session_id: Optional[str] = None) -> str:
    """Deterministic offline-safe default responder for the gateway.

    Provides predictable mock LLM completions without external dependencies or network calls.
    """
    return f"Echo [deterministic response]: {prompt}"


class GatewayService:
    """Core gateway coordinating guardrails, pluggable responder, and observability telemetry."""

    def __init__(
        self,
        responder: Optional[Callable[..., Any]] = None,
        store: Optional[ObservabilityStore] = None,
        max_input_length: int = 4000,
        max_output_length: int = 8000,
    ) -> None:
        self.responder = responder or default_responder
        self.store = store or default_store
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def _invoke_responder(self, prompt: str, session_id: Optional[str] = None) -> Any:
        """Call responder handling varying parameter counts."""
        try:
            sig = inspect.signature(self.responder)
            params = sig.parameters
            if len(params) >= 2 or any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values()):
                return self.responder(prompt, session_id)
            return self.responder(prompt)
        except (ValueError, TypeError):
            # Fallback if signature cannot be inspected
            try:
                return self.responder(prompt, session_id)
            except TypeError:
                return self.responder(prompt)

    def process(self, prompt: str, session_id: Optional[str] = None) -> GatewayResponse:
        """Process a request synchronously through input guardrails, responder, and output guardrails."""
        start_time = time.perf_counter()
        request_id = f"req_{uuid.uuid4().hex[:12]}"

        # 1. Run input guardrails & PII redaction
        input_result = check_input_guardrails(prompt, max_length=self.max_input_length)
        if not input_result.allowed:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            trace = RequestTrace(
                trace_id=request_id,
                session_id=session_id,
                prompt=input_result.text,
                response=None,
                status="blocked",
                allowed=False,
                latency_ms=latency_ms,
                flags=input_result.flags,
                redactions=input_result.redactions,
                reason=input_result.reason,
            )
            self.store.record(trace)
            return GatewayResponse(
                request_id=request_id,
                allowed=False,
                status="blocked",
                output=None,
                redacted_prompt=input_result.text,
                flags=input_result.flags,
                redactions=input_result.redactions,
                reason=input_result.reason,
                latency_ms=latency_ms,
                session_id=session_id,
            )

        # 2. Call pluggable responder
        try:
            res = self._invoke_responder(input_result.text, session_id)
            if inspect.iscoroutine(res):
                # If responder is a coroutine in sync execution, run in event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            raw_output = executor.submit(asyncio.run, res).result()
                    else:
                        raw_output = loop.run_until_complete(res)
                except RuntimeError:
                    raw_output = asyncio.run(res)
            else:
                raw_output = res
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            err_flags = list(dict.fromkeys(input_result.flags + ["RESPONDER_ERROR"]))
            reason = f"Responder execution failed: {exc}"
            trace = RequestTrace(
                trace_id=request_id,
                session_id=session_id,
                prompt=input_result.text,
                response=None,
                status="error",
                allowed=False,
                latency_ms=latency_ms,
                flags=err_flags,
                redactions=input_result.redactions,
                reason=reason,
            )
            self.store.record(trace)
            return GatewayResponse(
                request_id=request_id,
                allowed=False,
                status="error",
                output=None,
                redacted_prompt=input_result.text,
                flags=err_flags,
                redactions=input_result.redactions,
                reason=reason,
                latency_ms=latency_ms,
                session_id=session_id,
            )

        # 3. Run output guardrails & output PII redaction
        output_result = check_output_guardrails(str(raw_output), max_length=self.max_output_length)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        combined_flags = list(dict.fromkeys(input_result.flags + output_result.flags))
        combined_redactions = list(dict.fromkeys(input_result.redactions + output_result.redactions))

        if not output_result.allowed:
            trace = RequestTrace(
                trace_id=request_id,
                session_id=session_id,
                prompt=input_result.text,
                response=None,
                status="blocked",
                allowed=False,
                latency_ms=latency_ms,
                flags=combined_flags,
                redactions=combined_redactions,
                reason=output_result.reason,
            )
            self.store.record(trace)
            return GatewayResponse(
                request_id=request_id,
                allowed=False,
                status="blocked",
                output=None,
                redacted_prompt=input_result.text,
                flags=combined_flags,
                redactions=combined_redactions,
                reason=output_result.reason,
                latency_ms=latency_ms,
                session_id=session_id,
            )

        # 4. Request succeeded
        trace = RequestTrace(
            trace_id=request_id,
            session_id=session_id,
            prompt=input_result.text,
            response=output_result.text,
            status="allowed",
            allowed=True,
            latency_ms=latency_ms,
            flags=combined_flags,
            redactions=combined_redactions,
            reason="Success",
        )
        self.store.record(trace)
        return GatewayResponse(
            request_id=request_id,
            allowed=True,
            status="allowed",
            output=output_result.text,
            redacted_prompt=input_result.text,
            flags=combined_flags,
            redactions=combined_redactions,
            reason="Success",
            latency_ms=latency_ms,
            session_id=session_id,
        )

    async def aprocess(self, prompt: str, session_id: Optional[str] = None) -> GatewayResponse:
        """Process a request asynchronously through guardrails, responder, and telemetry."""
        start_time = time.perf_counter()
        request_id = f"req_{uuid.uuid4().hex[:12]}"

        # 1. Input guardrails
        input_result = check_input_guardrails(prompt, max_length=self.max_input_length)
        if not input_result.allowed:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            trace = RequestTrace(
                trace_id=request_id,
                session_id=session_id,
                prompt=input_result.text,
                response=None,
                status="blocked",
                allowed=False,
                latency_ms=latency_ms,
                flags=input_result.flags,
                redactions=input_result.redactions,
                reason=input_result.reason,
            )
            self.store.record(trace)
            return GatewayResponse(
                request_id=request_id,
                allowed=False,
                status="blocked",
                output=None,
                redacted_prompt=input_result.text,
                flags=input_result.flags,
                redactions=input_result.redactions,
                reason=input_result.reason,
                latency_ms=latency_ms,
                session_id=session_id,
            )

        # 2. Responder invocation
        try:
            res = self._invoke_responder(input_result.text, session_id)
            if inspect.iscoroutine(res):
                raw_output = await res
            else:
                raw_output = res
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            err_flags = list(dict.fromkeys(input_result.flags + ["RESPONDER_ERROR"]))
            reason = f"Responder execution failed: {exc}"
            trace = RequestTrace(
                trace_id=request_id,
                session_id=session_id,
                prompt=input_result.text,
                response=None,
                status="error",
                allowed=False,
                latency_ms=latency_ms,
                flags=err_flags,
                redactions=input_result.redactions,
                reason=reason,
            )
            self.store.record(trace)
            return GatewayResponse(
                request_id=request_id,
                allowed=False,
                status="error",
                output=None,
                redacted_prompt=input_result.text,
                flags=err_flags,
                redactions=input_result.redactions,
                reason=reason,
                latency_ms=latency_ms,
                session_id=session_id,
            )

        # 3. Output guardrails
        output_result = check_output_guardrails(str(raw_output), max_length=self.max_output_length)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        combined_flags = list(dict.fromkeys(input_result.flags + output_result.flags))
        combined_redactions = list(dict.fromkeys(input_result.redactions + output_result.redactions))

        if not output_result.allowed:
            trace = RequestTrace(
                trace_id=request_id,
                session_id=session_id,
                prompt=input_result.text,
                response=None,
                status="blocked",
                allowed=False,
                latency_ms=latency_ms,
                flags=combined_flags,
                redactions=combined_redactions,
                reason=output_result.reason,
            )
            self.store.record(trace)
            return GatewayResponse(
                request_id=request_id,
                allowed=False,
                status="blocked",
                output=None,
                redacted_prompt=input_result.text,
                flags=combined_flags,
                redactions=combined_redactions,
                reason=output_result.reason,
                latency_ms=latency_ms,
                session_id=session_id,
            )

        # 4. Success
        trace = RequestTrace(
            trace_id=request_id,
            session_id=session_id,
            prompt=input_result.text,
            response=output_result.text,
            status="allowed",
            allowed=True,
            latency_ms=latency_ms,
            flags=combined_flags,
            redactions=combined_redactions,
            reason="Success",
        )
        self.store.record(trace)
        return GatewayResponse(
            request_id=request_id,
            allowed=True,
            status="allowed",
            output=output_result.text,
            redacted_prompt=input_result.text,
            flags=combined_flags,
            redactions=combined_redactions,
            reason="Success",
            latency_ms=latency_ms,
            session_id=session_id,
        )
