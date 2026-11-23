"""Deterministic input and output guardrails for LLM Gateway."""

from __future__ import annotations

import re
from typing import List, Tuple
from pydantic import BaseModel, Field


class GuardrailResult(BaseModel):
    """Structured result returned by input and output guardrail evaluations."""

    allowed: bool
    reason: str
    flags: List[str] = Field(default_factory=list)
    redactions: List[str] = Field(default_factory=list)
    text: str


# ============================================================================
# PII Patterns and Redaction
# ============================================================================

EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_REGEX = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)

# Detect common API key signatures: OpenAI, GitHub, Google, Bearer / secret assignments
API_KEY_PATTERNS = [
    re.compile(r"\b(?:sk-[a-zA-Z0-9_\-]{20,}|gh[pousr]_[a-zA-Z0-9]{36,}|AIza[0-9A-Za-z_\-]{30,})\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|bearer)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-.]{16,})['\"]?"),
    re.compile(r"\b(?:bearer\s+)([a-zA-Z0-9_\-.]{20,})\b", re.IGNORECASE),
]


def redact_pii(text: str) -> Tuple[str, List[str]]:
    """Redact emails, phone numbers, and API-key-like tokens from text.

    Returns the redacted text and a list of redaction tag types detected.
    """
    redacted_text = text
    tags_detected: List[str] = []

    # 1. API Keys (redact first to prevent false phone/email matches inside tokens)
    api_key_found = False
    for pat in API_KEY_PATTERNS:
        matches = list(pat.finditer(redacted_text))
        if matches:
            api_key_found = True
            # If pattern has capturing group (e.g. key assignment), redact the token value
            for match in reversed(matches):
                if match.groups() and match.group(1):
                    start = match.start(1)
                    end = match.end(1)
                    redacted_text = redacted_text[:start] + "[REDACTED_API_KEY]" + redacted_text[end:]
                else:
                    start = match.start()
                    end = match.end()
                    redacted_text = redacted_text[:start] + "[REDACTED_API_KEY]" + redacted_text[end:]

    if api_key_found:
        tags_detected.append("API_KEY")

    # 2. Emails
    if EMAIL_REGEX.search(redacted_text):
        tags_detected.append("EMAIL")
        redacted_text = EMAIL_REGEX.sub("[REDACTED_EMAIL]", redacted_text)

    # 3. Phone Numbers
    if PHONE_REGEX.search(redacted_text):
        tags_detected.append("PHONE_NUMBER")
        redacted_text = PHONE_REGEX.sub("[REDACTED_PHONE]", redacted_text)

    return redacted_text, tags_detected


# ============================================================================
# Prompt Injection & Jailbreak Signatures
# ============================================================================

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bforget\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\boverride\s+(?:all\s+)?(?:previous|prior|system)\s+(?:rules|instructions|prompts)\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(?:dan|unrestricted|god\s+mode|developer\s+mode|jailbreak|jailbroken)\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(?:dan|an\s+unrestricted\s+ai|evil\s+ai|an\s+unfiltered\s+model)\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bdeveloper\s+mode\s+(?:is\s+)?enabled\b", re.IGNORECASE),
    re.compile(r"\bbypass\s+(?:all\s+)?(?:safety|content|ethical|system)\s+(?:filters|guardrails|policies|restrictions)\b", re.IGNORECASE),
    re.compile(r"\b(?:reveal|print|show|output|leak)\s+(?:your\s+)?(?:system\s+prompt|initial\s+instructions|system\s+instructions|base\s+prompt)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:are|were)\s+your\s+(?:initial|original|system)\s+instructions\b", re.IGNORECASE),
    # Delimiter injection
    re.compile(r"(?:<\|im_start\|>|<\|im_end\|>|<\|system\|>|\[INST\]|\[/INST\]|<<SYS>>|</SYS>)", re.IGNORECASE),
]

# Policy violation patterns (prohibited malicious topics)
POLICY_VIOLATION_PATTERNS = [
    re.compile(r"\bhow\s+to\s+(?:build|make|synthesize|create)\s+(?:a\s+bomb|explosives?|biological\s+weapon)\b", re.IGNORECASE),
    re.compile(r"\b(?:generate|write|code)\s+(?:malware|ransomware|keylogger|ddos\s+attack\s+script|trojan)\b", re.IGNORECASE),
    re.compile(r"\bhow\s+to\s+(?:hack\s+into|exploit|infiltrate)\s+(?:unauthorized|someone's)\b", re.IGNORECASE),
]


def detect_prompt_injection(text: str) -> bool:
    """Check if the text contains known prompt injection or jailbreak patterns."""
    return any(pattern.search(text) for pattern in PROMPT_INJECTION_PATTERNS)


def detect_policy_violation(text: str) -> bool:
    """Check if the text violates safety policy."""
    # Check null bytes or malicious binary payload characters
    if "\x00" in text:
        return True
    return any(pattern.search(text) for pattern in POLICY_VIOLATION_PATTERNS)


# ============================================================================
# Guardrail Validation Functions
# ============================================================================

def check_input_guardrails(
    prompt: str,
    max_length: int = 4000,
    min_length: int = 1,
    redact: bool = True,
) -> GuardrailResult:
    """Evaluate deterministic input guardrails on prompt text.

    Performs length check, prompt-injection / jailbreak detection,
    policy checks, and deterministic PII redaction.
    """
    # 1. Emptiness check
    stripped = prompt.strip()
    if len(stripped) < min_length:
        return GuardrailResult(
            allowed=False,
            reason="Prompt cannot be empty",
            flags=["EMPTY_PROMPT"],
            redactions=[],
            text=prompt,
        )

    # 2. Length check
    if len(prompt) > max_length:
        return GuardrailResult(
            allowed=False,
            reason=f"Prompt length ({len(prompt)}) exceeds maximum limit of {max_length} characters",
            flags=["LENGTH_EXCEEDED"],
            redactions=[],
            text=prompt,
        )

    # 3. Prompt injection & jailbreak detection
    if detect_prompt_injection(prompt):
        return GuardrailResult(
            allowed=False,
            reason="Prompt injection or jailbreak attempt detected",
            flags=["PROMPT_INJECTION"],
            redactions=[],
            text=prompt,
        )

    # 4. Policy checks
    if detect_policy_violation(prompt):
        return GuardrailResult(
            allowed=False,
            reason="Policy violation: prohibited content detected",
            flags=["POLICY_VIOLATION"],
            redactions=[],
            text=prompt,
        )

    # 5. PII Redaction
    flags: List[str] = []
    redacted_text = prompt
    redactions: List[str] = []

    if redact:
        redacted_text, redactions = redact_pii(prompt)
        if redactions:
            flags.append("PII_REDACTED")

    return GuardrailResult(
        allowed=True,
        reason="Passed input guardrails",
        flags=flags,
        redactions=redactions,
        text=redacted_text,
    )


def check_output_guardrails(
    text: str,
    max_length: int = 8000,
    redact: bool = True,
) -> GuardrailResult:
    """Evaluate deterministic output guardrails on LLM response text.

    Performs output length check, policy checks, and PII redaction.
    """
    # 1. Output length check
    if len(text) > max_length:
        return GuardrailResult(
            allowed=False,
            reason=f"Output length ({len(text)}) exceeds maximum limit of {max_length} characters",
            flags=["OUTPUT_LENGTH_EXCEEDED"],
            redactions=[],
            text=text,
        )

    # 2. Output policy violation check
    if detect_policy_violation(text):
        return GuardrailResult(
            allowed=False,
            reason="Policy violation detected in output",
            flags=["OUTPUT_POLICY_VIOLATION"],
            redactions=[],
            text=text,
        )

    # 3. PII Redaction on output (prevent leaking sensitive data in responses)
    flags: List[str] = []
    redacted_text = text
    redactions: List[str] = []

    if redact:
        redacted_text, redactions = redact_pii(text)
        if redactions:
            flags.append("PII_REDACTED")

    return GuardrailResult(
        allowed=True,
        reason="Passed output guardrails",
        flags=flags,
        redactions=redactions,
        text=redacted_text,
    )
