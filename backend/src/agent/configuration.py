import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent."""

    query_generator_model: str = Field(
        default="kimi-k3",
        metadata={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    reflection_model: str = Field(
        default="kimi-k3",
        metadata={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    answer_model: str = Field(
        default="kimi-k3",
        metadata={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "The number of initial search queries to generate."},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "The maximum number of research loops to perform."},
    )

    rag_enabled: bool = Field(
        default=True,
        metadata={"description": "Whether to enable RAG retrieval from the knowledge base."},
    )

    rag_top_k: int = Field(
        default=5,
        metadata={"description": "The number of top documents to retrieve from the knowledge base."},
    )

    docs_dir: str = Field(
        default="data/docs",
        metadata={"description": "Directory containing documents to index for RAG."},
    )

    chroma_persist_dir: str = Field(
        default="data/chroma",
        metadata={"description": "Directory to persist the Chroma vector store."},
    )

    # -----------------------------------------------------------------------
    # Tool-calling configuration (reserved capability)
    # -----------------------------------------------------------------------
    tool_calling_enabled: bool = Field(
        default=False,
        metadata={
            "description": (
                "Whether to use native LLM tool-calling / structured-output "
                "instead of manual JSON parsing."
            )
        },
    )

    # -----------------------------------------------------------------------
    # MCP (Model Context Protocol) configuration (reserved capability)
    # -----------------------------------------------------------------------
    mcp_enabled: bool = Field(
        default=False,
        metadata={
            "description": (
                "Whether to enable MCP (Model Context Protocol) tool servers. "
                "When enabled, tools from configured MCP servers are made "
                "available to the agent."
            )
        },
    )

    mcp_servers: str = Field(
        default="",
        metadata={
            "description": (
                "JSON-encoded list of MCPServerConfig objects. "
                'Example: \'[{"name":"filesystem","command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","/tmp"]}]\''
            )
        },
    )

    # -----------------------------------------------------------------------
    # PostgreSQL persistence configuration
    # -----------------------------------------------------------------------
    postgres_uri: str = Field(
        default="postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable",
        metadata={"description": "PostgreSQL connection URI for thread and memory persistence."},
    )

    memory_enabled: bool = Field(
        default=True,
        metadata={"description": "Whether to enable long-term memory storage across sessions."},
    )

    memory_compression_threshold: int = Field(
        default=10,
        metadata={"description": "Number of memory entries in a namespace before automatic compression triggers."},
    )

    memory_compression_batch_size: int = Field(
        default=10,
        metadata={"description": "Number of oldest entries to compress in one batch."},
    )

    hybrid_search_enabled: bool = Field(
        default=True,
        metadata={"description": "Whether to enable hybrid search (BM25 + vector) for RAG retrieval."},
    )

    bm25_enabled: bool = Field(
        default=True,
        metadata={"description": "Whether to enable BM25 keyword search in hybrid retrieval."},
    )

    rerank_enabled: bool = Field(
        default=True,
        metadata={"description": "Whether to enable cross-encoder reranking after hybrid retrieval."},
    )

    hybrid_search_top_k: int = Field(
        default=10,
        metadata={"description": "Initial top-k candidates to retrieve from each search modality before fusion."},
    )

    rerank_top_k: int = Field(
        default=5,
        metadata={"description": "Final top-k documents to return after reranking."},
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
