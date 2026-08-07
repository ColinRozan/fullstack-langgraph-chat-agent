"""Input security: sanitization, PII detection, and prompt-injection screening.

All functions are designed to be non-blocking (except *sanitize_input* which
only truncates / cleans).  PII detection and prompt-injection checks return
flags so the caller can decide whether to reject, log, or proceed.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

DEFAULT_MAX_LENGTH = int(os.environ.get("INPUT_MAX_LENGTH", "4000"))
PII_ENABLED = os.environ.get("PII_DETECTION_ENABLED", "true").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Dangerous / control characters
# ---------------------------------------------------------------------------

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# ---------------------------------------------------------------------------
# PII patterns (China-centric + general)
# ---------------------------------------------------------------------------

_PII_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "china_mobile",
        re.compile(
            r"(?<![\d])(?:1[3-9]\d{9})(?![\d])"
        ),
    ),
    (
        "china_id_card",
        re.compile(
            r"(?<![\d])[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?![\d])"
        ),
    ),
    (
        "email",
        re.compile(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        ),
    ),
    (
        "china_bank_card",
        re.compile(
            r"(?<![\d])(?:62\d{14,17}|4\d{15}|5[1-5]\d{14}|3[47]\d{13})(?![\d])"
        ),
    ),
]

# ---------------------------------------------------------------------------
# Prompt-injection heuristics
# ---------------------------------------------------------------------------

_INJECTION_KEYWORDS = [
    "ignore previous instructions",
    "ignore all prior instructions",
    "system prompt",
    "you are now",
    "new instructions",
    "disregard",
    "override",
    "leak",
    "prompt leak",
    "debug mode",
    "developer mode",
    " DAN ",
    "jailbreak",
]

_INJECTION_RE = re.compile(
    "|".join(re.escape(kw) for kw in _INJECTION_KEYWORDS),
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sanitize_input(text: str, max_length: int = DEFAULT_MAX_LENGTH) -> str:
    """Clean user input before sending to the LLM or storage.

    Steps:
        1. Strip outer whitespace.
        2. Remove control characters (except \n, \t, \r).
        3. Truncate to *max_length* with an ellipsis indicator.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = _CONTROL_CHAR_RE.sub("", text)
    if len(text) > max_length:
        text = text[: max_length - 3] + "..."
    return text


def detect_pii(text: str) -> Tuple[str, List[str]]:
    """Detect and mask PII in *text*.

    Returns:
        (sanitised_text, list_of_pii_types_detected)

    If ``PII_DETECTION_ENABLED`` is false, returns the original text and an
    empty list.
    """
    if not PII_ENABLED:
        return text, []

    detected: List[str] = []
    masked = text
    for pii_type, pattern in _PII_PATTERNS:
        found = pattern.findall(masked)
        if found:
            detected.append(pii_type)
            # Replace every match with a placeholder
            masked = pattern.sub(f"[{pii_type}_REDACTED]", masked)
    return masked, list(set(detected))


def check_prompt_injection(text: str) -> Tuple[bool, List[str]]:
    """Heuristic scan for potential prompt-injection payloads.

    Returns:
        (is_suspicious, matched_keywords)

    This is intentionally lightweight.  In a high-security environment,
    replace with a dedicated classifier (e.g. LLM Guard, Rebuff).
    """
    matches = _INJECTION_RE.findall(text)
    suspicious = len(matches) > 0
    return suspicious, list(set(m.lower() for m in matches))
