"""Core LangGraph agent definition with production hardening.

Includes structured logging, input sanitisation, token-cost tracking,
exponential-backoff retries, and observability metrics.
"""

import json
import os
import re

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from duckduckgo_search import DDGS
import requests

from agent.tools_and_schemas import (
    SearchQueryList,
    Reflection,
    ToolCallResult,
)
from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
    RagRetrieveState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    rag_reflection_instructions,
    answer_instructions,
)
from agent.utils import (
    get_research_topic,
    format_rag_documents,
    format_search_results,
)
from agent.knowledge_base import retrieve_documents
from agent.mcp_client import create_mcp_client, MCPClient
from agent import persistence
from agent.observability import (
    get_logger,
    set_trace_id,
    get_trace_id,
    SEARCH_REQUESTS_TOTAL,
    RAG_RETRIEVAL_DURATION,
    RESEARCH_LOOP_COUNT,
)
from agent.security import sanitize_input, detect_pii, check_prompt_injection
from agent.cost_tracking import record_llm_usage
from agent.retry_config import retry_with_backoff, with_circuit_breaker

load_dotenv()

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Checkpointer initialisation (lazy so import-time DB errors are avoided)
# ---------------------------------------------------------------------------
_checkpointer = None


def _get_checkpointer():
    """Return a PostgresSaver if available.

    In production we **do not** fall back to MemorySaver silently because that
    would lose data.  Only fall back when ``DB_FALLBACK_ENABLED`` is true.
    """
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer
    try:
        saver = persistence.get_postgres_saver()
        if saver is not None:
            _checkpointer = saver
            logger.info("checkpointer_postgres_ready")
            return _checkpointer
    except Exception as e:
        logger.warning("checkpointer_postgres_unavailable", error=str(e))

    if os.environ.get("DB_FALLBACK_ENABLED", "false").lower() in ("1", "true", "yes"):
        from langgraph.checkpoint.memory import MemorySaver

        _checkpointer = MemorySaver()
        logger.warning("checkpointer_memory_fallback")
        return _checkpointer

    raise RuntimeError(
        "PostgreSQL checkpointer is unavailable and DB_FALLBACK_ENABLED is false."
    )


# ---------------------------------------------------------------------------
# LLM wrappers with retry, circuit breaker, and cost tracking
# ---------------------------------------------------------------------------

class SimpleOpenAIChat(BaseChatModel):
    """Lightweight OpenAI-compatible chat model using requests.

    Includes production retries, configurable timeout, and automatic token
    / cost telemetry.
    """

    model: str
    temperature: float = 0
    max_retries: int = 2
    max_tokens: int = 8192
    timeout: int = 60

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": _convert_role(m), "content": str(m.content)}
                for m in messages
            ],
        }
        if stop:
            payload["stop"] = stop

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                msg = AIMessage(content=content)
                gen = ChatGeneration(message=msg)
                return ChatResult(generations=[gen])
            except Exception as e:
                logger.warning(
                    "openai_api_call_failed",
                    attempt=attempt,
                    max_retries=self.max_retries,
                    error=str(e),
                )
                if attempt == self.max_retries:
                    raise RuntimeError(f"OpenAI API call failed: {e}") from e

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "simple_openai_chat"


def _convert_role(msg: BaseMessage) -> str:
    if isinstance(msg, SystemMessage):
        return "system"
    if isinstance(msg, AIMessage):
        return "assistant"
    return "user"


def _get_llm(model: str, temperature: float = 0, max_tokens: int = 8192):
    """Create an LLM instance with production hardening.

    Uses OpenAI-compatible API if OPENAI_API_KEY is set, otherwise falls back
    to ChatAnthropic for Ark.
    """
    timeout = int(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_BASE_URL"):
        return SimpleOpenAIChat(
            model=model,
            temperature=temperature,
            max_retries=2,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    # Fallback to Anthropic / Ark
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_retries=2,
        max_tokens=max_tokens,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL"),
        timeout=timeout,
    )


def _extract_text(content) -> str:
    """Normalize LLM message content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _extract_json(text: str) -> dict:
    """Extract JSON object from LLM output text."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    raise ValueError("No JSON object found in LLM output")


# ---------------------------------------------------------------------------
# Tool-calling helpers (reserved capability)
# ---------------------------------------------------------------------------

def _is_tool_calling_supported(llm: BaseChatModel) -> bool:
    return hasattr(llm, "bind_tools") and hasattr(llm, "with_structured_output")


def _create_structured_llm(llm: BaseChatModel, schema, use_tool_calling: bool = False):
    if use_tool_calling and _is_tool_calling_supported(llm):
        try:
            return llm.with_structured_output(schema, include_raw=False)
        except Exception as e:
            logger.warning(
                "structured_output_fallback",
                error=str(e),
            )
    return _ManualJsonParser(llm, schema)


class _ManualJsonParser:
    """Fallback wrapper that invokes the LLM in plain-text mode and parses JSON manually."""

    def __init__(self, llm: BaseChatModel, schema):
        self.llm = llm
        self.schema = schema

    def invoke(self, prompt, **kwargs):
        raw = self.llm.invoke(prompt, **kwargs).content
        raw_text = _extract_text(raw)
        try:
            parsed = _extract_json(raw_text)
            return self.schema(**parsed)
        except Exception as e:
            logger.error(
                "json_parse_failed",
                error=str(e),
                raw_preview=raw_text[:200],
            )
            raise


# ---------------------------------------------------------------------------
# MCP helpers (reserved capability)
# ---------------------------------------------------------------------------

_mcp_client_singleton: MCPClient | None = None


def _get_or_create_mcp_client(configurable: Configuration) -> MCPClient | None:
    if not configurable.mcp_enabled:
        return None
    if _mcp_client_singleton is None and configurable.mcp_servers:
        _mcp_client_singleton = create_mcp_client(configurable.mcp_servers)
    return _mcp_client_singleton


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question."""
    configurable = Configuration.from_runnable_config(config)

    # Propagate trace ID through LangGraph config metadata
    trace_id = config.get("metadata", {}).get("trace_id") if config else None
    if trace_id:
        set_trace_id(trace_id)

    # Input sanitisation
    raw_topic = get_research_topic(state["messages"])
    topic, pii_types = detect_pii(sanitize_input(raw_topic, max_length=configurable.input_max_length))
    is_injection, injection_kws = check_prompt_injection(raw_topic)
    if pii_types:
        logger.info("pii_detected_in_input", types=pii_types)
    if is_injection:
        logger.warning("prompt_injection_detected", keywords=injection_kws)

    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    llm = _get_llm(configurable.query_generator_model, temperature=1.0)

    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=topic,
        number_queries=state["initial_search_query_count"],
    )

    structured_llm = _create_structured_llm(
        llm, SearchQueryList, use_tool_calling=configurable.tool_calling_enabled
    )
    try:
        result = structured_llm.invoke(formatted_prompt)
    except Exception as e:
        logger.error("generate_query_failed", error=str(e))
        result = SearchQueryList(
            query=[topic],
            rationale="Fallback single query due to generation failure.",
        )

    # Cost tracking
    if configurable.cost_tracking_enabled:
        record_llm_usage(
            model=configurable.query_generator_model,
            stage="generate_query",
            prompt_text=formatted_prompt,
            completion_text=str(result.query),
        )

    return {
        "search_query": result.query,
        "research_topic": topic,
    }


def continue_to_research(state: QueryGenerationState):
    """Spawn parallel web research nodes and a single RAG retrieval node."""
    sends = [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]
    if state.get("research_topic"):
        sends.append(
            Send("rag_retrieve", {"query": state["research_topic"], "id": "rag"})
        )
    return sends


def rag_retrieve(state: RagRetrieveState, config: RunnableConfig) -> OverallState:
    """Retrieve relevant documents from the local knowledge base."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.rag_enabled:
        return {"rag_documents": []}

    import time
    start = time.perf_counter()
    try:
        docs = retrieve_documents(
            state["query"],
            top_k=configurable.rag_top_k,
            use_hybrid=configurable.hybrid_search_enabled,
            enable_bm25=configurable.bm25_enabled,
            enable_rerank=configurable.rerank_enabled,
            hybrid_top_k=configurable.hybrid_search_top_k,
        )
        RAG_RETRIEVAL_DURATION.observe(time.perf_counter() - start)
        return {"rag_documents": docs}
    except Exception as exc:
        logger.warning("rag_retrieval_failed", error=str(exc))
        RAG_RETRIEVAL_DURATION.observe(time.perf_counter() - start)
        return {"rag_documents": []}


def _searx_search(query: str, max_results: int = 5):
    """Fallback search using public SearXNG instances."""
    instances = [
        "https://search.sapti.me",
        "https://searx.be",
        "https://search.bus-hit.me",
    ]
    for base in instances:
        try:
            resp = requests.get(
                f"{base}/search",
                params={"q": query, "format": "json", "engines": "bing,google"},
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                return [
                    {"title": r.get("title", ""), "href": r.get("url", ""), "body": r.get("content", "")}
                    for r in results[:max_results]
                ]
        except Exception:
            continue
    return []


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """Perform web research using DuckDuckGo / SearX and LLM."""
    configurable = Configuration.from_runnable_config(config)
    query = state["search_query"]

    # Perform web search: try DuckDuckGo with retries across backends
    search_results = []
    backends = ["api", "html"]
    for backend in backends:
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=5, backend=backend)
                search_results = list(results)
                logger.info(
                    "web_search_ddgs",
                    backend=backend,
                    result_count=len(search_results),
                    query=query,
                )
                SEARCH_REQUESTS_TOTAL.labels(provider=f"ddgs_{backend}", status="success").inc()
                if search_results:
                    break
        except Exception as e:
            logger.warning(
                "web_search_ddgs_failed",
                backend=backend,
                query=query,
                error=str(e),
            )
            SEARCH_REQUESTS_TOTAL.labels(provider=f"ddgs_{backend}", status="error").inc()

    # Fallback to SearX if DDGS yields nothing
    if not search_results:
        search_results = _searx_search(query, max_results=5)
        logger.info(
            "web_search_searx",
            result_count=len(search_results),
            query=query,
        )
        SEARCH_REQUESTS_TOTAL.labels(
            provider="searx",
            status="success" if search_results else "error",
        ).inc()

    # Format search results for the LLM
    formatted_search = format_search_results(search_results, state["id"])

    # Build prompt with search results
    current_date = get_current_date()
    formatted_prompt = web_searcher_instructions.format(
        current_date=current_date,
        research_topic=query,
        search_results=formatted_search,
    )

    # Use LLM to synthesize the search results
    llm = _get_llm(configurable.query_generator_model, temperature=0)
    try:
        result = llm.invoke(formatted_prompt)
        synthesized_text = _extract_text(result.content)
    except Exception as e:
        logger.error("web_research_llm_failed", query=query, error=str(e))
        synthesized_text = f"Error synthesizing search results for '{query}'."

    # Cost tracking
    if configurable.cost_tracking_enabled:
        record_llm_usage(
            model=configurable.query_generator_model,
            stage="web_research",
            prompt_text=formatted_prompt,
            completion_text=synthesized_text,
        )

    # Build simplified sources from search results
    sources_gathered = []
    for i, r in enumerate(search_results):
        sources_gathered.append(
            {
                "label": r.get("title", "Source"),
                "short_url": f"https://ddg.id/{state['id']}-{i}",
                "value": r.get("href", ""),
            }
        )

    return {
        "sources_gathered": sources_gathered,
        "search_query": [query],
        "web_research_result": [synthesized_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """Identify knowledge gaps and generate potential follow-up queries."""
    configurable = Configuration.from_runnable_config(config)
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    current_date = get_current_date()
    research_topic = get_research_topic(state["messages"])
    summaries = "\n\n---\n\n".join(str(item) for item in state["web_research_result"])
    summaries = summaries[:4000]

    rag_docs = state.get("rag_documents", [])
    if rag_docs:
        formatted_prompt = rag_reflection_instructions.format(
            current_date=current_date,
            research_topic=research_topic,
            summaries=summaries,
            rag_documents=format_rag_documents(rag_docs),
        )
    else:
        formatted_prompt = reflection_instructions.format(
            current_date=current_date,
            research_topic=research_topic,
            summaries=summaries,
        )

    llm = _get_llm(reasoning_model, temperature=0)
    structured_llm = _create_structured_llm(
        llm, Reflection, use_tool_calling=configurable.tool_calling_enabled
    )
    try:
        result = structured_llm.invoke(formatted_prompt)
    except Exception as e:
        logger.error("reflection_failed", error=str(e))
        result = Reflection(
            is_sufficient=True,
            knowledge_gap="",
            follow_up_queries=[],
        )

    # Cost tracking
    if configurable.cost_tracking_enabled:
        record_llm_usage(
            model=reasoning_model,
            stage="reflection",
            prompt_text=formatted_prompt,
            completion_text=str(result.dict()),
        )

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """Routing function that determines the next step in the research flow."""
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )

    RESEARCH_LOOP_COUNT.observe(state["research_loop_count"])

    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


# ---------------------------------------------------------------------------
# Optional agent nodes with tool-calling support (reserved capability)
# ---------------------------------------------------------------------------

def agent_with_tools(state: OverallState, config: RunnableConfig) -> OverallState:
    """Optional LangGraph node that lets the LLM call tools (MCP or bound tools)."""
    configurable = Configuration.from_runnable_config(config)
    llm = _get_llm(configurable.answer_model, temperature=0)

    tools = []
    mcp_client = _get_or_create_mcp_client(configurable)
    if mcp_client:
        try:
            tools.extend(mcp_client.to_langchain_tools())
        except Exception as e:
            logger.warning("mcp_tools_load_failed", error=str(e))

    if tools and _is_tool_calling_supported(llm):
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm

    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def execute_tool_calls(state: OverallState, config: RunnableConfig) -> OverallState:
    """Execute any tool calls present in the last AIMessage."""
    configurable = Configuration.from_runnable_config(config)
    messages = state["messages"]
    if not messages:
        return {"messages": []}

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not getattr(last_message, "tool_calls", None):
        return {"messages": []}

    tool_messages = []
    mcp_client = _get_or_create_mcp_client(configurable)

    for tc in last_message.tool_calls:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool_id = tc.get("id", "")

        output = None
        error = None

        if mcp_client:
            try:
                import asyncio

                output = asyncio.run(mcp_client.call_tool(tool_name, tool_args))
            except Exception as e:
                error = f"MCP tool call failed: {e}"

        if output is None and error is None:
            error = f"Tool '{tool_name}' is not available via MCP and no local handler is registered."

        tool_messages.append(
            ToolMessage(
                content=json.dumps({"output": output, "error": error}, default=str),
                tool_call_id=tool_id,
                name=tool_name,
            )
        )

    return {"messages": tool_messages}


# ---------------------------------------------------------------------------
# Finalize answer
# ---------------------------------------------------------------------------

def finalize_answer(state: OverallState, config: RunnableConfig):
    """Synthesize all gathered information into a coherent, cited answer.

    Also persists session metadata, long-term memory, and audit logs.
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    current_date = get_current_date()
    rag_docs = state.get("rag_documents", [])
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(str(item) for item in state["web_research_result"]),
        rag_documents=format_rag_documents(rag_docs),
    )

    llm = _get_llm(reasoning_model, temperature=0)
    try:
        result = llm.invoke(formatted_prompt)
        answer_text = _extract_text(result.content)
    except Exception as e:
        logger.error("finalize_answer_llm_failed", error=str(e))
        answer_text = "I encountered an error while generating the final answer. Please try again."

    # Cost tracking for final answer
    total_tokens = 0
    total_cost = 0.0
    if configurable.cost_tracking_enabled:
        pt, ct, cost = record_llm_usage(
            model=reasoning_model,
            stage="finalize_answer",
            prompt_text=formatted_prompt,
            completion_text=answer_text,
        )
        total_tokens = pt + ct
        total_cost = cost

    # Replace short urls with original urls
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in answer_text:
            answer_text = answer_text.replace(source["short_url"], source["value"])
            unique_sources.append(source)

    # Build structured RAG sources for frontend display
    rag_sources = []
    for i, doc in enumerate(rag_docs, 1):
        source = doc.metadata.get("source", "unknown") if hasattr(doc, "metadata") else "unknown"
        page = doc.metadata.get("page", "") if hasattr(doc, "metadata") else ""
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        rag_sources.append({
            "index": i,
            "source": source,
            "page": page,
            "preview": content[:300] + "..." if len(content) > 300 else content,
        })

    # -----------------------------------------------------------------------
    # Persist session metadata + long-term memory + audit
    # -----------------------------------------------------------------------
    try:
        thread_id = config.get("configurable", {}).get("thread_id") if config else None
        research_topic = get_research_topic(state["messages"])

        if thread_id:
            title = research_topic[:60] + "..." if len(research_topic) > 60 else research_topic
            persistence.upsert_thread(thread_id, title)

            persistence.save_session_state(
                thread_id,
                {
                    "research_topic": research_topic,
                    "answer_preview": answer_text[:500],
                    "sources_count": len(unique_sources),
                    "rag_sources_count": len(rag_sources),
                    "research_loop_count": state.get("research_loop_count", 0),
                },
            )

        # Cross-session memory
        if configurable.memory_enabled and research_topic:
            persistence.put_memory(
                "research_topics",
                research_topic[:128],
                {
                    "topic": research_topic,
                    "last_answer_preview": answer_text[:500],
                    "sources_count": len(unique_sources),
                    "timestamp": current_date,
                },
            )

            # Automatic memory compression
            try:
                from agent.memory_compression import maybe_compress_memory

                llm_compress = _get_llm(configurable.answer_model, temperature=0)
                maybe_compress_memory(
                    "research_topics",
                    llm=llm_compress,
                    threshold=configurable.memory_compression_threshold,
                    batch_size=configurable.memory_compression_batch_size,
                )
            except Exception as compress_err:
                logger.warning("memory_compression_failed", error=str(compress_err))

        # Update audit log with response metadata
        if thread_id:
            persistence.update_audit_log_response(
                thread_id=thread_id,
                response_summary=answer_text[:200],
                tokens_used=total_tokens or None,
                cost_usd=total_cost or None,
            )
    except Exception as e:
        logger.warning("finalize_persistence_failed", error=str(e))

    return {
        "messages": [AIMessage(content=answer_text)],
        "sources_gathered": unique_sources,
        "rag_sources": rag_sources,
    }


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------

builder = StateGraph(OverallState, config_schema=Configuration)

builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("rag_retrieve", rag_retrieve)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Reserved-capability nodes (not wired by default)
builder.add_node("agent_with_tools", agent_with_tools)
builder.add_node("execute_tool_calls", execute_tool_calls)

builder.add_edge(START, "generate_query")
builder.add_conditional_edges(
    "generate_query", continue_to_research, ["web_research", "rag_retrieve"]
)
builder.add_edge("web_research", "reflection")
builder.add_edge("rag_retrieve", "reflection")
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
builder.add_edge("finalize_answer", END)

graph = builder.compile(
    name="pro-search-agent",
)
