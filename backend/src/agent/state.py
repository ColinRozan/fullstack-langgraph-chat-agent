from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, Any

from langgraph.graph import add_messages
from typing_extensions import Annotated


import operator


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    rag_documents: Annotated[list, operator.add]
    rag_sources: list
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str
    # ------------------------------------------------------------------
    # Reserved fields for tool-calling and MCP capabilities
    # ------------------------------------------------------------------
    tool_calls: Annotated[list, operator.add]
    mcp_available_tools: list
    mcp_context: Any


class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    research_loop_count: int
    number_of_ran_queries: int


class Query(TypedDict):
    query: str
    rationale: str


class QueryGenerationState(TypedDict):
    search_query: list[Query]
    research_topic: str


class WebSearchState(TypedDict):
    search_query: str
    id: str


class RagRetrieveState(TypedDict):
    query: str
    id: str


@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str = field(default=None)  # Final report
