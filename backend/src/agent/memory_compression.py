"""Memory compression using LLM summarization.

When a namespace accumulates more entries than a threshold, this module
compresses the oldest batch into a concise summary, reducing storage bloat
while preserving key facts.
"""

import json
import time
from typing import Any, Dict

from langchain_core.language_models.chat_models import BaseChatModel

from agent import persistence


_COMPRESSION_PROMPT_TEMPLATE = """\
You are a memory compression assistant. Your task is to compress a set of memory entries into a concise summary that preserves all key facts, decisions, and context.

Original memories:
{memories}

Please output a JSON object with the following structure (no markdown fences):
{{
  "summary": "A concise paragraph summarizing the key information from all memories.",
  "key_facts": ["fact 1", "fact 2", "fact 3"]
}}
"""


def _extract_json(text: str) -> dict:
    """Extract JSON object from text."""
    import re
    # Try fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Try raw JSON object
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    raise ValueError("No JSON object found in LLM output")


def maybe_compress_memory(
    namespace: str,
    llm: BaseChatModel,
    threshold: int = 10,
    batch_size: int = 10,
) -> bool:
    """Compress memories in a namespace if they exceed threshold.

    This function checks the number of (non-compressed) entries in the given
    namespace. If the count is >= *threshold*, the oldest *batch_size* entries
    are fetched, summarized by the provided LLM, and replaced with a single
    compressed memory entry.

    Args:
        namespace: The memory namespace to evaluate.
        llm: A LangChain chat model used to generate the compression summary.
        threshold: Minimum number of entries required to trigger compression.
        batch_size: Number of oldest entries to compress in one batch.

    Returns:
        True if compression was performed, False otherwise.
    """
    try:
        count = persistence.count_memory(namespace)
    except Exception as e:
        print(f"[memory_compression] Failed to count memory for '{namespace}': {e}")
        return False

    if count < threshold:
        return False

    # Fetch oldest memories for compression
    try:
        memories = persistence.list_memory_namespace(
            namespace, limit=batch_size, order_by="created_at ASC"
        )
    except Exception as e:
        print(f"[memory_compression] Failed to list memory for '{namespace}': {e}")
        return False

    if not memories:
        return False

    # Format memories for the LLM prompt
    formatted_parts = []
    for k, v in memories.items():
        formatted_parts.append(f"- Key: {k}\n  Value: {json.dumps(v, ensure_ascii=False)}")
    formatted_memories = "\n\n".join(formatted_parts)

    prompt = _COMPRESSION_PROMPT_TEMPLATE.format(memories=formatted_memories)

    try:
        response = llm.invoke(prompt)
        raw_text = response.content if hasattr(response, "content") else str(response)
        parsed = _extract_json(raw_text)
    except Exception as e:
        print(f"[memory_compression] LLM compression failed for '{namespace}': {e}")
        return False

    summary = parsed.get("summary", "")
    key_facts = parsed.get("key_facts", [])
    original_count = len(memories)

    compressed_value: Dict[str, Any] = {
        "_is_compressed": True,
        "summary": summary,
        "key_facts": key_facts,
        "original_count": original_count,
        "original_keys": list(memories.keys()),
        "compressed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    compressed_key = f"_compressed_{int(time.time() * 1000)}"

    try:
        persistence.put_memory(namespace, compressed_key, compressed_value)
        deleted = persistence.delete_memory_batch(namespace, list(memories.keys()))
        print(
            f"[memory_compression] Compressed {deleted} entries in '{namespace}' "
            f"into key '{compressed_key}'"
        )
    except Exception as e:
        print(f"[memory_compression] Failed to persist compression for '{namespace}': {e}")
        return False

    return True
