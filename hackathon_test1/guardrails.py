"""
guardrails.py – Agri Doctor Guardrail Engine
=============================================
Implements all nine guardrail categories for the Agri Doctor multi-agent
application. Every check is pure-Python (no external API) so it runs
instantly with zero extra cost.

Categories:
  1. Input Guardrails        – validate user-supplied text & images
  2. Output Guardrails       – validate AI-generated responses
  3. Tool Execution          – control agent→API interaction
  4. Data Guardrails         – data privacy & masking
  5. Security Guardrails     – injection, jailbreak, credential leaks
  6. Operational Guardrails  – tokens, timing, rate-limits, retries
  7. Human-in-the-Loop       – escalation triggers
  8. Multi-Agent Guardrails  – cross-agent safety
  9. Ethical & Compliance    – bias, fairness, audit trail
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Shared data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GuardResult:
    """Returned by every guardrail check."""
    passed: bool
    category: str
    check: str
    message: str = ""
    severity: str = "INFO"        # INFO | WARNING | BLOCK
    details: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# 1. INPUT GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────

# Prompt-injection patterns (attempts to override system instructions)
_INJECTION_PATTERNS = [
    r"ignore\s+.{0,30}(instructions?|prompt|rules?)",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(a\s+)?(?!farmer|expert|pathologist)",  # allow legitimate roles
    r"disregard\s+.{0,20}(guidelines?|instructions?|rules?)",
    r"new\s+instructions?\s*:",
    r"override\s+(mode|safety|guardrails?)",
    r"do\s+not\s+follow\s+.{0,20}(rules?|guidelines?|instructions?)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
]

# Clearly malicious / off-topic inputs
_MALICIOUS_PATTERNS = [
    r"<script[\s>]",
    r"javascript\s*:",
    r"on(load|error|click)\s*=",
    r"(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE)\s+\w+",  # SQL
    r"(__import__|exec\s*\(|eval\s*\(|os\.system)",            # code injection
]

# PII patterns
_PII_PATTERNS = {
    "email":        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    "phone":        r"\b(\+?\d[\d\s\-().]{7,}\d)\b",
    "credit_card":  r"\b(?:\d[ -]?){13,16}\b",
    "national_id":  r"\b[A-Z]{1,2}\d{6,9}\b",
    "passport":     r"\b[A-Z]{1,2}\d{7,9}\b",
}

# Inappropriate / abusive content keywords (abbreviated – expand as needed)
_INAPPROPRIATE_KEYWORDS = [
    "hate", "kill", "murder", "suicide", "bomb", "explosive",
    "drug", "cocaine", "heroin", "methamphetamine",
]

# Maximum allowed text-input length (characters)
MAX_INPUT_CHARS = 2000
# Maximum allowed image size (bytes) – 10 MB
MAX_IMAGE_BYTES = 10 * 1024 * 1024
# Allowed MIME types
ALLOWED_MIME_TYPES = {"image/jpeg", "image/jpg", "image/png"}


def check_prompt_injection(text: str) -> GuardResult:
    """Detect attempts to override system instructions."""
    lower = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lower):
            return GuardResult(
                passed=False,
                category="Input",
                check="PromptInjection",
                severity="BLOCK",
                message="Input contains a potential prompt-injection attempt.",
                details={"pattern": pattern},
            )
    return GuardResult(passed=True, category="Input", check="PromptInjection")


def check_malicious_input(text: str) -> GuardResult:
    """Detect script injection, SQL injection, and code execution attempts."""
    for pattern in _MALICIOUS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardResult(
                passed=False,
                category="Input",
                check="MaliciousInput",
                severity="BLOCK",
                message="Input contains potentially malicious code or commands.",
                details={"pattern": pattern},
            )
    return GuardResult(passed=True, category="Input", check="MaliciousInput")


def check_input_length(text: str) -> GuardResult:
    """Ensure input does not exceed the maximum allowed length."""
    if len(text) > MAX_INPUT_CHARS:
        return GuardResult(
            passed=False,
            category="Input",
            check="InputLength",
            severity="BLOCK",
            message=f"Input exceeds maximum allowed length of {MAX_INPUT_CHARS} characters.",
            details={"length": len(text)},
        )
    return GuardResult(passed=True, category="Input", check="InputLength")


def check_pii(text: str) -> GuardResult:
    """Detect personally identifiable information in free-text inputs."""
    found: dict[str, list[str]] = {}
    for label, pattern in _PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            found[label] = matches
    if found:
        return GuardResult(
            passed=False,
            category="Input",
            check="PIIDetection",
            severity="WARNING",
            message="Input may contain personally identifiable information (PII).",
            details={"pii_types": list(found.keys())},
        )
    return GuardResult(passed=True, category="Input", check="PIIDetection")


def check_inappropriate_content(text: str) -> GuardResult:
    """Filter clearly inappropriate or abusive keywords."""
    lower = text.lower()
    found = [kw for kw in _INAPPROPRIATE_KEYWORDS if kw in lower]
    if found:
        return GuardResult(
            passed=False,
            category="Input",
            check="InappropriateContent",
            severity="BLOCK",
            message="Input contains inappropriate or harmful content.",
            details={"keywords": found},
        )
    return GuardResult(passed=True, category="Input", check="InappropriateContent")


def check_image_upload(image_bytes: bytes, mime_type: str) -> GuardResult:
    """Validate uploaded image: size and MIME type."""
    if mime_type not in ALLOWED_MIME_TYPES:
        return GuardResult(
            passed=False,
            category="Input",
            check="ImageValidation",
            severity="BLOCK",
            message=f"Unsupported image type '{mime_type}'. Allowed: JPEG, PNG.",
        )
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return GuardResult(
            passed=False,
            category="Input",
            check="ImageValidation",
            severity="BLOCK",
            message=f"Image exceeds maximum size of {MAX_IMAGE_BYTES // (1024*1024)} MB.",
            details={"size_bytes": len(image_bytes)},
        )
    return GuardResult(passed=True, category="Input", check="ImageValidation")


def run_input_guardrails(
    text: str = "",
    image_bytes: bytes | None = None,
    mime_type: str = "",
) -> list[GuardResult]:
    """Run all input guardrail checks and return results."""
    results: list[GuardResult] = []
    if text:
        results += [
            check_prompt_injection(text),
            check_malicious_input(text),
            check_input_length(text),
            check_pii(text),
            check_inappropriate_content(text),
        ]
    if image_bytes is not None:
        results.append(check_image_upload(image_bytes, mime_type))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. OUTPUT GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────

_HALLUCINATION_PHRASES = [
    r"as of my knowledge cutoff",
    r"i cannot verify",
    r"i don[''`]t have access to real[\s-]time",
    r"i made up",
    r"fictitious",
    r"imaginary",
]

_TOXIC_PATTERNS = [
    r"\b(idiot|stupid|fool|moron|imbecile|bastard|bitch|asshole|shit|damn|crap)\b",
]

_SENSITIVE_OUTPUT_PATTERNS = {
    "api_key":    r"(?i)(api[_\s-]?key|secret[_\s-]?key)\s*[:=]\s*\S+",
    "password":   r"(?i)password\s*[:=]\s*\S+",
    "token":      r"(?i)(access[_\s-]?token|bearer)\s*[:=]\s*\S+",
    "private_key": r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
}

# Minimum reasonable response length (characters)
MIN_RESPONSE_CHARS = 50


def check_hallucination(text: str) -> GuardResult:
    """Flag responses that appear to self-report uncertainty or fabrication."""
    lower = text.lower()
    for pattern in _HALLUCINATION_PHRASES:
        if re.search(pattern, lower):
            return GuardResult(
                passed=False,
                category="Output",
                check="HallucinationDetection",
                severity="WARNING",
                message="Response may contain uncertain or unverifiable claims.",
                details={"pattern": pattern},
            )
    return GuardResult(passed=True, category="Output", check="HallucinationDetection")


def check_factual_accuracy(text: str) -> GuardResult:
    """
    Lightweight factual sanity check: ensure the response mentions
    crop/plant/disease-related terms for an agricultural context.
    """
    agriculture_terms = [
        "disease", "plant", "crop", "leaf", "fungus", "bacteria", "virus",
        "symptom", "treatment", "fertilizer", "pesticide", "pathogen",
        "infection", "blight", "mildew", "rust", "rot", "wilt",
    ]
    lower = text.lower()
    found = [t for t in agriculture_terms if t in lower]
    if not found:
        return GuardResult(
            passed=False,
            category="Output",
            check="FactualAccuracy",
            severity="WARNING",
            message="Response does not appear to address agricultural / crop-disease topics.",
        )
    return GuardResult(passed=True, category="Output", check="FactualAccuracy")


def check_toxicity(text: str) -> GuardResult:
    """Filter toxic or harmful language in AI output."""
    lower = text.lower()
    for pattern in _TOXIC_PATTERNS:
        if re.search(pattern, lower):
            return GuardResult(
                passed=False,
                category="Output",
                check="ToxicityFilter",
                severity="BLOCK",
                message="Response contains potentially harmful or offensive language.",
            )
    return GuardResult(passed=True, category="Output", check="ToxicityFilter")


def check_sensitive_output(text: str) -> GuardResult:
    """Ensure AI output does not leak credentials or secrets."""
    for label, pattern in _SENSITIVE_OUTPUT_PATTERNS.items():
        if re.search(pattern, text):
            return GuardResult(
                passed=False,
                category="Output",
                check="SensitiveInfoFilter",
                severity="BLOCK",
                message=f"Response may contain sensitive information ({label}).",
            )
    return GuardResult(passed=True, category="Output", check="SensitiveInfoFilter")


def check_response_format(text: str) -> GuardResult:
    """Ensure response is non-empty and meets minimum length."""
    stripped = text.strip()
    if not stripped:
        return GuardResult(
            passed=False,
            category="Output",
            check="ResponseFormat",
            severity="BLOCK",
            message="Empty response received from the model.",
        )
    if len(stripped) < MIN_RESPONSE_CHARS:
        return GuardResult(
            passed=False,
            category="Output",
            check="ResponseFormat",
            severity="WARNING",
            message="Response is unusually short and may be incomplete.",
            details={"length": len(stripped)},
        )
    return GuardResult(passed=True, category="Output", check="ResponseFormat")


def run_output_guardrails(text: str) -> list[GuardResult]:
    """Run all output guardrail checks."""
    return [
        check_hallucination(text),
        check_factual_accuracy(text),
        check_toxicity(text),
        check_sensitive_output(text),
        check_response_format(text),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 3. TOOL EXECUTION GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────

# Whitelist of permitted IBM watsonx tool-call model IDs
_ALLOWED_MODELS = {
    "mistralai/mistral-small-3-1-24b-instruct-2503",
    "mistralai/mistral-medium-2505",
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    "ibm/granite-4-h-small",
    "ibm/granite-3-3-8b-instruct",
}

_ALLOWED_ENDPOINTS = {
    "https://us-south.ml.cloud.ibm.com",
    "https://eu-de.ml.cloud.ibm.com",
    "https://eu-gb.ml.cloud.ibm.com",
    "https://jp-tok.ml.cloud.ibm.com",
}


def check_model_access(model_id: str) -> GuardResult:
    """Verify the requested model is on the approved list."""
    if model_id not in _ALLOWED_MODELS:
        return GuardResult(
            passed=False,
            category="ToolExecution",
            check="ModelAccessControl",
            severity="WARNING",
            message=f"Model '{model_id}' is not on the approved list. Proceeding with caution.",
            details={"model_id": model_id, "allowed": sorted(_ALLOWED_MODELS)},
        )
    return GuardResult(passed=True, category="ToolExecution", check="ModelAccessControl")


def check_endpoint_authorization(url: str) -> GuardResult:
    """Ensure API calls are directed to approved IBM endpoints."""
    # Normalise – strip trailing slash
    normalized = url.rstrip("/")
    if normalized not in _ALLOWED_ENDPOINTS:
        return GuardResult(
            passed=False,
            category="ToolExecution",
            check="EndpointAuthorization",
            severity="BLOCK",
            message=f"Endpoint '{url}' is not in the authorised list.",
        )
    return GuardResult(passed=True, category="ToolExecution", check="EndpointAuthorization")


def check_function_call(function_name: str, allowed_functions: set[str]) -> GuardResult:
    """Validate that an agent-requested function is permitted."""
    if function_name not in allowed_functions:
        return GuardResult(
            passed=False,
            category="ToolExecution",
            check="FunctionCallValidation",
            severity="BLOCK",
            message=f"Function '{function_name}' is not authorised for execution.",
        )
    return GuardResult(passed=True, category="ToolExecution", check="FunctionCallValidation")


def run_tool_guardrails(model_id: str, endpoint_url: str) -> list[GuardResult]:
    """Run tool-execution guardrail checks."""
    return [
        check_model_access(model_id),
        check_endpoint_authorization(endpoint_url),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────

def mask_pii(text: str) -> str:
    """Replace detected PII with masked placeholders."""
    result = text
    for label, pattern in _PII_PATTERNS.items():
        result = re.sub(pattern, f"[{label.upper()}_REDACTED]", result)
    return result


def check_data_privacy(text: str) -> GuardResult:
    """Detect and report PII before it is sent to an external API."""
    pii_result = check_pii(text)
    if not pii_result.passed:
        return GuardResult(
            passed=False,
            category="Data",
            check="DataPrivacy",
            severity="WARNING",
            message="Text contains PII – it will be masked before being sent to the AI.",
            details=pii_result.details,
        )
    return GuardResult(passed=True, category="Data", check="DataPrivacy")


def compute_image_hash(image_bytes: bytes) -> str:
    """Return a SHA-256 fingerprint of the image (for audit, not storage)."""
    return hashlib.sha256(image_bytes).hexdigest()


def run_data_guardrails(text: str = "", image_bytes: bytes | None = None) -> list[GuardResult]:
    """Run data-protection guardrail checks."""
    results = [check_data_privacy(text)] if text else []
    if image_bytes:
        # Log image hash to audit trail; don't block.
        img_hash = compute_image_hash(image_bytes)
        results.append(GuardResult(
            passed=True,
            category="Data",
            check="ImageFingerprint",
            severity="INFO",
            message="Image fingerprinted for audit trail.",
            details={"sha256": img_hash[:16] + "…"},
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. SECURITY GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────

_JAILBREAK_PATTERNS = [
    r"ignore\s+.{0,20}(safety|ethical|content)\s+(filter|policy|guardrail)",
    r"(disable|bypass|remove)\s+.{0,20}(restriction|filter|guardrail|safety)",
    r"you\s+have\s+no\s+(restriction|limitation|rule)",
    r"do\s+anything\s+now",
    r"evil\s+(ai|bot|assistant|mode)",
    r"without\s+(any\s+)?(restriction|ethical|moral|filter)",
]

_CREDENTIAL_LEAK_PATTERNS = [
    r"(?i)api[_\s-]?key\s*[:=]\s*[A-Za-z0-9\-_]{20,}",
    r"(?i)bearer\s+[A-Za-z0-9\-_.]{20,}",
    r"(?i)eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+",  # JWT
]


def check_jailbreak(text: str) -> GuardResult:
    """Detect jailbreak or safety-bypass attempts."""
    lower = text.lower()
    for pattern in _JAILBREAK_PATTERNS:
        if re.search(pattern, lower):
            return GuardResult(
                passed=False,
                category="Security",
                check="JailbreakDetection",
                severity="BLOCK",
                message="Input appears to be a jailbreak or safety-bypass attempt.",
                details={"pattern": pattern},
            )
    return GuardResult(passed=True, category="Security", check="JailbreakDetection")


def check_credential_exposure(text: str) -> GuardResult:
    """Ensure no credentials or tokens appear in user input or AI output."""
    for pattern in _CREDENTIAL_LEAK_PATTERNS:
        if re.search(pattern, text):
            return GuardResult(
                passed=False,
                category="Security",
                check="CredentialExposure",
                severity="BLOCK",
                message="Text contains what appears to be an API key, token, or credential.",
            )
    return GuardResult(passed=True, category="Security", check="CredentialExposure")


def run_security_guardrails(text: str) -> list[GuardResult]:
    """Run security guardrail checks."""
    return [
        check_jailbreak(text),
        check_credential_exposure(text),
        check_malicious_input(text),   # also covers SQL/XSS injection
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 6. OPERATIONAL GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────

# Global in-memory rate-limit store {session_id: [timestamps]}
_rate_limit_store: dict[str, list[float]] = {}

MAX_REQUESTS_PER_MINUTE = 10
MAX_TOKENS_PER_REQUEST = 2048
MAX_EXECUTION_SECONDS = 120
MAX_AGENT_ITERATIONS = 5


def check_token_limit(requested_tokens: int) -> GuardResult:
    """Enforce a per-request token ceiling."""
    if requested_tokens > MAX_TOKENS_PER_REQUEST:
        return GuardResult(
            passed=False,
            category="Operational",
            check="TokenLimit",
            severity="WARNING",
            message=f"Requested {requested_tokens} tokens exceeds limit of {MAX_TOKENS_PER_REQUEST}. Capped automatically.",
            details={"requested": requested_tokens, "limit": MAX_TOKENS_PER_REQUEST},
        )
    return GuardResult(passed=True, category="Operational", check="TokenLimit")


def check_rate_limit(session_id: str) -> GuardResult:
    """Sliding-window rate limiter: max N requests per 60 seconds."""
    now = time.time()
    window = 60.0
    timestamps = _rate_limit_store.get(session_id, [])
    # Keep only timestamps within the last window
    timestamps = [t for t in timestamps if now - t < window]
    if len(timestamps) >= MAX_REQUESTS_PER_MINUTE:
        return GuardResult(
            passed=False,
            category="Operational",
            check="RateLimit",
            severity="BLOCK",
            message=f"Rate limit exceeded: max {MAX_REQUESTS_PER_MINUTE} requests per minute.",
            details={"requests_in_window": len(timestamps)},
        )
    timestamps.append(now)
    _rate_limit_store[session_id] = timestamps
    return GuardResult(passed=True, category="Operational", check="RateLimit")


def check_agent_iterations(iteration: int) -> GuardResult:
    """Prevent runaway agent loops."""
    if iteration > MAX_AGENT_ITERATIONS:
        return GuardResult(
            passed=False,
            category="Operational",
            check="AgentIterationLimit",
            severity="BLOCK",
            message=f"Agent has exceeded the maximum iteration count ({MAX_AGENT_ITERATIONS}).",
            details={"iteration": iteration},
        )
    return GuardResult(passed=True, category="Operational", check="AgentIterationLimit")


def run_operational_guardrails(
    session_id: str,
    requested_tokens: int,
    iteration: int = 1,
) -> list[GuardResult]:
    """Run operational guardrail checks."""
    return [
        check_token_limit(requested_tokens),
        check_rate_limit(session_id),
        check_agent_iterations(iteration),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 7. HUMAN-IN-THE-LOOP GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────

# Phrases that warrant human expert escalation
_ESCALATION_TRIGGERS = [
    r"\blow\b.*\bconfidence\b",
    r"\bconfidence\b.*\blow\b",
    r"unclear\s+image",
    r"insufficient\s+(image|data|information)",
    r"cannot\s+(identify|determine|diagnose)",
    r"recommend\s+(laboratory|lab)\s+(test|analysis|confirmation)",
    r"consult\s+(a\s+)?(expert|specialist|pathologist|extension)",
    r"seek\s+(professional|expert)\s+(advice|help|assistance)",
    r"severe\s+(infection|outbreak|damage)",
    r"widespread\s+(disease|infection)",
]


def check_escalation_needed(text: str) -> GuardResult:
    """Detect if a diagnosis requires human expert escalation."""
    lower = text.lower()
    triggers_found = []
    for pattern in _ESCALATION_TRIGGERS:
        if re.search(pattern, lower):
            triggers_found.append(pattern)
    if triggers_found:
        return GuardResult(
            passed=False,   # "failed" = escalation IS needed
            category="HumanInTheLoop",
            check="EscalationTrigger",
            severity="WARNING",
            message="Diagnosis indicates uncertainty or severity – human expert review recommended.",
            details={"triggers": triggers_found[:3]},
        )
    return GuardResult(passed=True, category="HumanInTheLoop", check="EscalationTrigger")


def run_hitl_guardrails(agent_output: str) -> list[GuardResult]:
    """Run human-in-the-loop guardrail checks on agent output."""
    return [check_escalation_needed(agent_output)]


# ─────────────────────────────────────────────────────────────────────────────
# 8. MULTI-AGENT GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────

# Permitted agent-to-agent communication roles
_PERMITTED_AGENT_ROLES = {"pathologist", "extension_officer"}
# Maximum output length passed between agents (characters)
MAX_INTER_AGENT_CHARS = 8000


def check_agent_role(role: str) -> GuardResult:
    """Verify the calling agent has an authorised role."""
    if role.lower() not in _PERMITTED_AGENT_ROLES:
        return GuardResult(
            passed=False,
            category="MultiAgent",
            check="AgentRoleControl",
            severity="BLOCK",
            message=f"Agent role '{role}' is not authorised.",
            details={"allowed_roles": sorted(_PERMITTED_AGENT_ROLES)},
        )
    return GuardResult(passed=True, category="MultiAgent", check="AgentRoleControl")


def check_inter_agent_payload(payload: str, source_role: str, target_role: str) -> GuardResult:
    """
    Validate the payload passed from one agent to another:
    – not empty, not oversized, and passes security checks.
    """
    if not payload.strip():
        return GuardResult(
            passed=False,
            category="MultiAgent",
            check="InterAgentPayload",
            severity="BLOCK",
            message=f"Empty payload from '{source_role}' to '{target_role}'.",
        )
    if len(payload) > MAX_INTER_AGENT_CHARS:
        return GuardResult(
            passed=False,
            category="MultiAgent",
            check="InterAgentPayload",
            severity="WARNING",
            message=f"Inter-agent payload exceeds {MAX_INTER_AGENT_CHARS} characters and will be truncated.",
            details={"length": len(payload)},
        )
    # Run security check on the payload too
    sec = check_credential_exposure(payload)
    if not sec.passed:
        return GuardResult(
            passed=False,
            category="MultiAgent",
            check="InterAgentPayload",
            severity="BLOCK",
            message="Inter-agent payload contains sensitive data and cannot be forwarded.",
        )
    return GuardResult(passed=True, category="MultiAgent", check="InterAgentPayload")


def check_loop_prevention(iteration: int) -> GuardResult:
    """Prevent infinite agent loops by capping iterations."""
    return check_agent_iterations(iteration)  # reuse Operational check


def run_multi_agent_guardrails(
    source_role: str,
    target_role: str,
    payload: str,
    iteration: int = 1,
) -> list[GuardResult]:
    """Run multi-agent guardrail checks."""
    return [
        check_agent_role(source_role),
        check_agent_role(target_role),
        check_inter_agent_payload(payload, source_role, target_role),
        check_loop_prevention(iteration),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 9. ETHICAL & COMPLIANCE GUARDRAILS
# ─────────────────────────────────────────────────────────────────────────────

# Terms that might indicate geographic / demographic bias
_BIAS_INDICATORS = [
    r"\b(only\s+)?(western|developed|first[\s-]world)\s+(country|countries|farms?)\b",
    r"\b(poor|primitive|backward)\s+(farmer|village|country)\b",
    r"\ball\s+(african|asian|indian|african)\s+farmers?\b",
]


def check_bias(text: str) -> GuardResult:
    """Detect potentially biased generalisations in AI output."""
    lower = text.lower()
    for pattern in _BIAS_INDICATORS:
        if re.search(pattern, lower):
            return GuardResult(
                passed=False,
                category="Ethical",
                check="BiasDetection",
                severity="WARNING",
                message="Response may contain geographic or demographic bias.",
                details={"pattern": pattern},
            )
    return GuardResult(passed=True, category="Ethical", check="BiasDetection")


def check_disclaimer_present(text: str) -> GuardResult:
    """Ensure AI output contains required responsibility disclaimer."""
    disclaimer_terms = [
        "not a confirmed", "consult", "laboratory", "professional",
        "extension officer", "expert", "ai-generated", "preliminary",
    ]
    lower = text.lower()
    found = [t for t in disclaimer_terms if t in lower]
    if not found:
        return GuardResult(
            passed=False,
            category="Ethical",
            check="DisclaimerPresent",
            severity="WARNING",
            message="Response may be missing a responsibility disclaimer.",
        )
    return GuardResult(passed=True, category="Ethical", check="DisclaimerPresent")


def run_ethical_guardrails(text: str) -> list[GuardResult]:
    """Run ethical and compliance guardrail checks."""
    return [
        check_bias(text),
        check_disclaimer_present(text),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Audit Trail
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    agent: str = ""
    stage: str = ""
    results: list[GuardResult] = field(default_factory=list)

    def blocked(self) -> bool:
        return any(r.severity == "BLOCK" and not r.passed for r in self.results)

    def warnings(self) -> list[GuardResult]:
        return [r for r in self.results if r.severity == "WARNING" and not r.passed]

    def summary(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "agent": self.agent,
            "stage": self.stage,
            "blocked": self.blocked(),
            "warnings": len(self.warnings()),
            "checks": len(self.results),
        }


# Module-level audit log (in-memory for the session)
_audit_log: list[AuditEntry] = []


def record_audit(entry: AuditEntry) -> None:
    _audit_log.append(entry)


def get_audit_log() -> list[AuditEntry]:
    return list(_audit_log)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run ALL guardrails for a given stage
# ─────────────────────────────────────────────────────────────────────────────

def run_all_input_checks(
    session_id: str,
    text: str = "",
    image_bytes: bytes | None = None,
    mime_type: str = "",
    requested_tokens: int = 1024,
    vision_model_id: str = "",
    endpoint_url: str = "https://us-south.ml.cloud.ibm.com",
) -> AuditEntry:
    """
    Master input guardrail runner. Call before invoking Agent 1.
    Returns an AuditEntry; call .blocked() to decide whether to abort.
    """
    results: list[GuardResult] = []
    results += run_input_guardrails(text, image_bytes, mime_type)
    results += run_security_guardrails(text)
    results += run_data_guardrails(text, image_bytes)
    results += run_tool_guardrails(vision_model_id, endpoint_url)
    results += run_operational_guardrails(session_id, requested_tokens, iteration=1)
    entry = AuditEntry(agent="input", stage="pre_agent1", results=results)
    record_audit(entry)
    return entry


def run_all_agent1_output_checks(output: str) -> AuditEntry:
    """Run output + multi-agent guardrails on Agent 1's response."""
    results: list[GuardResult] = []
    results += run_output_guardrails(output)
    results += run_security_guardrails(output)
    results += run_hitl_guardrails(output)
    results += run_ethical_guardrails(output)
    # Validate inter-agent handoff
    results += run_multi_agent_guardrails(
        source_role="pathologist",
        target_role="extension_officer",
        payload=output,
        iteration=2,
    )
    entry = AuditEntry(agent="pathologist", stage="post_agent1", results=results)
    record_audit(entry)
    return entry


def run_all_agent2_output_checks(output: str) -> AuditEntry:
    """Run output + ethical guardrails on Agent 2's final advisory."""
    results: list[GuardResult] = []
    results += run_output_guardrails(output)
    results += run_security_guardrails(output)
    results += run_ethical_guardrails(output)
    results += run_hitl_guardrails(output)
    entry = AuditEntry(agent="extension_officer", stage="post_agent2", results=results)
    record_audit(entry)
    return entry
