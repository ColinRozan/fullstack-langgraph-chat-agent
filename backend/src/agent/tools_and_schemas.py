from typing import List, Optional, Any
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


# ---------------------------------------------------------------------------
# Tool-calling schemas (reserved capability)
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    name: str = Field(description="The name of the tool to call.")
    arguments: dict = Field(description="Arguments to pass to the tool.")


class ToolCallRequest(BaseModel):
    tool_calls: List[ToolCall] = Field(description="List of tool calls to execute.")


class ToolCallResult(BaseModel):
    name: str = Field(description="The name of the tool that was called.")
    arguments: dict = Field(description="The arguments passed to the tool.")
    output: Any = Field(description="The output returned by the tool.")
    error: Optional[str] = Field(default=None, description="Error message if the tool call failed.")


# ---------------------------------------------------------------------------
# MCP (Model Context Protocol) schemas (reserved capability)
# ---------------------------------------------------------------------------

class MCPServerConfig(BaseModel):
    name: str = Field(description="Human-readable name for the MCP server.")
    command: Optional[str] = Field(default=None, description="Command to launch the MCP server (stdio transport).")
    args: Optional[List[str]] = Field(default=None, description="Arguments for the command.")
    url: Optional[str] = Field(default=None, description="URL for SSE transport.")
    env: Optional[dict] = Field(default=None, description="Environment variables for the server process.")


class MCPToolDefinition(BaseModel):
    name: str = Field(description="The name of the tool exposed by the MCP server.")
    description: str = Field(description="A description of what the tool does.")
    input_schema: dict = Field(description="JSONSchema describing the tool's input parameters.")
    server_name: Optional[str] = Field(default=None, description="The MCP server that exposes this tool.")
