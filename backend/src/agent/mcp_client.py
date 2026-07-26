"""MCP (Model Context Protocol) client integration — reserved capability.

This module provides optional MCP support for the agent. When ``mcp_enabled``
is ``True`` in the configuration, the agent attempts to connect to the
configured MCP servers, discover their tools, and make those tools available
to the LLM.

If ``langchain-mcp-adapters`` is not installed or MCP is disabled, all
functions gracefully degrade to no-ops so the existing research flow is
unaffected.
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

from agent.tools_and_schemas import MCPServerConfig, MCPToolDefinition

try:
    from langchain_core.tools import StructuredTool
    _HAS_LANGCHAIN = True
except Exception:  # pragma: no cover
    _HAS_LANGCHAIN = False

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _HAS_MCP = True
except Exception:  # pragma: no cover
    _HAS_MCP = False


def _load_mcp_server_configs(raw_json: str) -> List[MCPServerConfig]:
    """Parse the JSON-encoded MCP server configuration string."""
    if not raw_json or not raw_json.strip():
        return []
    try:
        data = json.loads(raw_json)
        if not isinstance(data, list):
            return []
        return [MCPServerConfig(**item) for item in data]
    except Exception as e:
        print(f"[MCP] Failed to parse mcp_servers config: {e}")
        return []


class MCPClient:
    """Lightweight MCP client that discovers and invokes tools from MCP servers.

    This is a **reserved capability**. When ``mcp_enabled`` is ``False`` (the
    default), the client does nothing and the agent behaves exactly as before.
    """

    def __init__(self, server_configs: List[MCPServerConfig]):
        self.server_configs = server_configs
        self._tools: List[MCPToolDefinition] = []
        self._sessions: List[Any] = []
        self._exit_stacks: List[Any] = []

    @property
    def tools(self) -> List[MCPToolDefinition]:
        """List of tools discovered from connected MCP servers."""
        return self._tools

    async def connect(self) -> None:
        """Connect to all configured MCP servers and discover tools."""
        if not _HAS_MCP:
            print("[MCP] mcp package not installed. Skipping MCP connection.")
            return
        if not self.server_configs:
            return

        for cfg in self.server_configs:
            try:
                await self._connect_server(cfg)
            except Exception as e:
                print(f"[MCP] Failed to connect to server '{cfg.name}': {e}")

    async def _connect_server(self, cfg: MCPServerConfig) -> None:
        """Connect to a single MCP server via stdio transport."""
        import contextlib

        if not cfg.command:
            return

        server_params = StdioServerParameters(
            command=cfg.command,
            args=cfg.args or [],
            env={**os.environ, **(cfg.env or {})},
        )

        ctx = stdio_client(server_params)
        read_stream, write_stream = await ctx.__aenter__()
        self._exit_stacks.append(ctx)

        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()
        self._sessions.append(session)

        await session.initialize()
        tools_result = await session.list_tools()

        for tool in tools_result.tools:
            self._tools.append(
                MCPToolDefinition(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                    server_name=cfg.name,
                )
            )
        print(f"[MCP] Connected to '{cfg.name}' — discovered {len(tools_result.tools)} tool(s).")

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """Call an MCP tool by name with the given arguments."""
        if not _HAS_MCP or not self._sessions:
            return {"error": "MCP not available"}

        for session in self._sessions:
            try:
                result = await session.call_tool(tool_name, arguments=arguments)
                return result
            except Exception:
                continue
        return {"error": f"Tool '{tool_name}' not found on any connected MCP server."}

    async def disconnect(self) -> None:
        """Cleanly disconnect from all MCP servers."""
        for session in self._sessions:
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
        for ctx in self._exit_stacks:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self._sessions.clear()
        self._exit_stacks.clear()

    def to_langchain_tools(self) -> List[Any]:
        """Convert discovered MCP tools to LangChain ``StructuredTool`` objects.

        Returns an empty list if ``langchain-mcp-adapters`` is not installed.
        """
        if not _HAS_LANGCHAIN:
            return []

        lc_tools: List[Any] = []
        for t in self._tools:
            # Build a minimal function signature from the JSONSchema
            # so StructuredTool can render a useful description.
            schema = t.input_schema

            async def _invoke(**kwargs) -> Any:
                return await self.call_tool(t.name, kwargs)

            lc_tools.append(
                StructuredTool.from_function(
                    name=t.name,
                    description=f"[{t.server_name}] {t.description}",
                    func=_invoke,
                    coroutine=_invoke,
                    args_schema=None,  # StructuredTool infers from func signature
                )
            )
        return lc_tools


def create_mcp_client(raw_config_json: str) -> MCPClient:
    """Factory that creates an :class:`MCPClient` from a raw JSON config string."""
    configs = _load_mcp_server_configs(raw_config_json)
    return MCPClient(configs)
