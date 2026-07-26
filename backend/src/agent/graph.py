import json
import os
import re

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
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

load_dotenv()

class SimpleOpenAIChat(BaseChatModel):
    """Lightweight OpenAI-compatible chat model using requests."""

    model: str
    temperature: float = 0
    max_retries: int = 2
    max_tokens: int = 8192

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
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                msg = AIMessage(content=content)
                gen = ChatGeneration(message=msg)
                return ChatResult(generations=[gen])
            except Exception as e:
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
    """Create an LLM instance. Uses OpenAI-compatible API if OPENAI_API_KEY is set,
    otherwise falls back to ChatAnthropic for Ark."""
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_BASE_URL"):
        return SimpleOpenAIChat(
            model=model,
            temperature=temperature,
            max_retries=2,
            max_tokens=max_tokens,
        )
    # Fallback to Anthropic / Ark
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_retries=2,
        max_tokens=max_tokens,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        anthropic_api_url=os.getenv("ANTHROPIC_BASE_URL"),
    )


def _extract_text(content) -> str:
    """Normalize LLM message content to a plain string.

    Anthropic-compatible APIs may return content as a list of dicts
    (e.g. [{"type": "text", "text": "..."}]). This helper flattens
    everything into a single string.
    """
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
    """Extract JSON object from LLM output text.

    Tries to find a JSON block inside triple backticks first,
    then falls back to the first `{...}` substring.
    """
    # Try fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Try raw JSON object
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    raise ValueError("No JSON object found in LLM output")


# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses Claude to create optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init LLM
    llm = _get_llm(configurable.query_generator_model, temperature=1.0)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries (plain text + manual JSON parse)
    raw = llm.invoke(formatted_prompt).content
    try:
        parsed = _extract_json(raw)
        result = SearchQueryList(**parsed)
    except Exception as e:
        print(f"[generate_query] Failed to parse JSON: {e}. Raw output:\n{raw}")
        # Fallback: return a single query derived from the research topic
        result = SearchQueryList(
            query=[get_research_topic(state["messages"])],
            rationale="Fallback single query due to parsing failure.",
        )
    return {
        "search_query": result.query,
        "research_topic": get_research_topic(state["messages"]),
    }


def continue_to_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node
    and the user's research topic to the RAG retrieval node.

    This spawns parallel web research nodes (one per query) and a single
    RAG retrieval node.
    """
    sends = [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]
    # Add RAG retrieval in parallel if enabled
    if state.get("research_topic"):
        sends.append(
            Send("rag_retrieve", {"query": state["research_topic"], "id": "rag"})
        )
    return sends


def rag_retrieve(state: RagRetrieveState, config: RunnableConfig) -> OverallState:
    """LangGraph node that retrieves relevant documents from the local knowledge base.

    Uses Chroma vector store to find the top-k most relevant document chunks
    for the user's query.

    Args:
        state: Current graph state containing the research topic/query
        config: Configuration for the runnable, including RAG settings

    Returns:
        Dictionary with state update, including rag_documents
    """
    configurable = Configuration.from_runnable_config(config)
    if not configurable.rag_enabled:
        return {"rag_documents": []}

    try:
        docs = retrieve_documents(state["query"], top_k=configurable.rag_top_k)
        return {"rag_documents": docs}
    except Exception:
        # Gracefully degrade if RAG fails
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
                timeout=10,
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
    """LangGraph node that performs web research using DuckDuckGo / SearX and LLM.

    Executes a web search, then uses LLM to synthesize the search results
    into a coherent summary with source citations.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    configurable = Configuration.from_runnable_config(config)
    query = state["search_query"]

    # Perform web search: try DuckDuckGo with retries across backends
    search_results = []
    backends = ["api", "html"]  # api/auto is generally more reliable; html as fallback
    for backend in backends:
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=5, backend=backend)
                search_results = list(results)
                print(f"[Web Research] DDGS({backend}) returned {len(search_results)} results for '{query}'")
                if search_results:
                    break
        except Exception as e:
            print(f"[Web Research] DDGS({backend}) failed for '{query}': {e}")

    # Fallback to SearX if DDGS yields nothing
    if not search_results:
        search_results = _searx_search(query, max_results=5)
        print(f"[Web Research] SearX fallback returned {len(search_results)} results for '{query}'")

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
        print(f"[web_research] LLM failed for query '{query}': {e}")
        synthesized_text = f"Error synthesizing search results for '{query}'."

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
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary and knowledge base documents to identify areas
    for further research and generates potential follow-up queries.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt (truncate summaries to avoid overly long prompts)
    current_date = get_current_date()
    research_topic = get_research_topic(state["messages"])
    summaries = "\n\n---\n\n".join(str(item) for item in state["web_research_result"])
    # Truncate to ~4000 chars to keep prompt size reasonable for Ark
    summaries = summaries[:4000]

    # Choose prompt based on whether we have RAG documents
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

    # init Reasoning Model (plain text + manual JSON parse)
    llm = _get_llm(reasoning_model, temperature=0)
    try:
        raw = llm.invoke(formatted_prompt).content
    except Exception as e:
        print(f"[reflection] LLM failed: {e}")
        # Fallback: declare sufficient and stop the loop
        return {
            "is_sufficient": True,
            "knowledge_gap": "",
            "follow_up_queries": [],
            "research_loop_count": state["research_loop_count"],
            "number_of_ran_queries": len(state["search_query"]),
        }

    try:
        parsed = _extract_json(raw)
        result = Reflection(**parsed)
    except Exception as e:
        print(f"[reflection] Failed to parse JSON: {e}. Raw output:\n{raw}")
        # Fallback: declare sufficient and stop the loop
        result = Reflection(
            is_sufficient=True,
            knowledge_gap="",
            follow_up_queries=[],
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
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
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


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining web research and knowledge base documents to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    rag_docs = state.get("rag_documents", [])
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(str(item) for item in state["web_research_result"]),
        rag_documents=format_rag_documents(rag_docs),
    )

    # init Reasoning Model, default to Claude 3.5 Sonnet
    llm = _get_llm(reasoning_model, temperature=0)
    try:
        result = llm.invoke(formatted_prompt)
        answer_text = _extract_text(result.content)
    except Exception as e:
        print(f"[finalize_answer] LLM failed: {e}")
        answer_text = "I encountered an error while generating the final answer. Please try again."

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in answer_text:
            answer_text = answer_text.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=answer_text)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("rag_retrieve", rag_retrieve)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to spawn parallel web research and RAG retrieval
builder.add_conditional_edges(
    "generate_query", continue_to_research, ["web_research", "rag_retrieve"]
)
# Both web_research and rag_retrieve feed into reflection
builder.add_edge("web_research", "reflection")
builder.add_edge("rag_retrieve", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
