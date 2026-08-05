# Fullstack LangGraph Quickstart

A fullstack research assistant powered by **LangGraph**, **LLM APIs**, and a modern **React** frontend. The agent performs comprehensive research on your queries by dynamically generating search terms, querying the web via DuckDuckGo (with SearXNG fallback), retrieving documents from a local knowledge base, reflecting on results to identify knowledge gaps, and iteratively refining its search until it delivers a well-supported answer with inline citations.

<img src="./2.png" title="Fullstack LangGraph" alt="Fullstack LangGraph" width="90%">

<img src="./3.png" title="Fullstack LangGraph" alt="Fullstack LangGraph" width="90%">

<img src="./4.png" title="Fullstack LangGraph" alt="Fullstack LangGraph" width="90%">

## Features

- 💬 **Conversational Research UI** — Modern React chat interface with real-time streaming.
- 🧠 **LangGraph Agent** — Stateful agent workflow with query generation, parallel web research, RAG retrieval, reflection, and answer synthesis.
- 🗂️ **Multi-Thread Memory** — Sidebar shows all historical conversations. Each conversation is an independent thread with full state persistence; switch threads without losing context.
- 🐘 **PostgreSQL Persistence** — Thread metadata, session snapshots, and long-term memory are persisted to PostgreSQL. LangGraph checkpointing uses `PostgresSaver` (falls back to `MemorySaver` when the DB is unavailable).
- 🧠 **Cross-Session Long-Term Memory** — Automatically stores research topics and answer previews in a namespace key-value store across conversations.
- 🗜️ **Automatic Memory Compression** — When a memory namespace exceeds a configurable threshold, oldest entries are automatically summarized and compressed to prevent unbounded growth.
- 🔍 **Dynamic Query Generation** — LLM generates diverse, targeted search queries from your question.
- 🌐 **Multi-Backend Web Search** — DuckDuckGo primary search with automatic SearXNG fallback.
- 📚 **Hybrid RAG Knowledge Base** — Local document retrieval with hybrid search (BM25 keyword + vector semantic + cross-encoder reranking) for significantly better recall; supports PDF, TXT, and Markdown.
- 🤔 **Reflective Reasoning** — Analyzes gathered information to identify gaps and decides whether to continue researching.
- 📄 **Inline Citations** — Distinguishes web sources `[🌐 Title](URL)` from knowledge base sources `[📄 source: filename.pdf]`.
- 🎯 **Research Depth Control** — Choose between Low, Medium, and High effort modes to adjust query count and max reflection loops.
- 🔄 **Model Selection** — Switch between available LLM models for different agent stages.
- 🎨 **Modern Dark UI** — Tailwind CSS + Shadcn UI with collapsible activity timeline showing each research step live.
- 🔧 **Native Tool-Calling** (reserved) — Optional native LLM structured-output / tool-calling for query generation and reflection (falls back to manual JSON parsing when disabled).
- 🔌 **MCP Support** (reserved) — Pluggable Model Context Protocol server integration for external tool discovery and invocation.
- 💾 **PDF Export** — Export AI answers as print-ready PDFs with light-theme rendering.
- 🐳 **Docker Ready** — Multi-stage Dockerfile and `docker-compose.yml` for production deployment with Redis and PostgreSQL.

## Project Structure

```
.
├── frontend/                     # React + Vite frontend application
│   ├── src/
│   │   ├── App.tsx              # Main app: thread manager + chat session
│   │   ├── components/
│   │   │   ├── ChatMessagesView.tsx   # Chat history, markdown rendering, sources
│   │   │   ├── InputForm.tsx          # Text input with effort/model selectors
│   │   │   ├── WelcomeScreen.tsx      # Landing page
│   │   │   ├── ActivityTimeline.tsx   # Real-time research step timeline
│   │   │   ├── Sidebar.tsx            # Thread history sidebar (new/delete/switch)
│   │   │   └── ui/                    # Shadcn UI primitives
│   │   ├── lib/
│   │   │   └── api.ts                 # Frontend API client for thread & memory endpoints
│   │   ├── main.tsx
│   │   └── global.css
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
├── backend/                      # LangGraph + FastAPI backend
│   ├── src/agent/
│   │   ├── graph.py             # Core LangGraph agent definition (nodes & edges)
│   │   ├── state.py             # TypedDict state definitions
│   │   ├── prompts.py           # LLM prompt templates
│   │   ├── configuration.py     # Agent configuration schema
│   │   ├── knowledge_base.py    # RAG retrieval with Chroma vector store + hybrid search support
│   │   ├── hybrid_retriever.py  # Hybrid retriever: BM25 + vector fusion + cross-encoder rerank
│   │   ├── memory_compression.py # Automatic memory compression when thresholds are exceeded
│   │   ├── tools_and_schemas.py # Pydantic schemas for structured LLM outputs & tool/MCP schemas
│   │   ├── mcp_client.py        # MCP (Model Context Protocol) client — reserved capability
│   │   ├── persistence.py       # PostgreSQL persistence layer (thread metadata, memory, checkpointer)
│   │   ├── utils.py             # Helper utilities
│   │   └── app.py               # FastAPI entry point (serves frontend at /app)
│   ├── examples/
│   │   └── cli_research.py      # CLI example for running research
│   ├── data/
│   │   ├── docs/                # Place your documents here for RAG
│   │   └── chroma/              # Chroma vector store persistence
│   ├── pyproject.toml
│   ├── langgraph.json           # LangGraph project configuration
│   └── .env.example
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── README.md
```

## Getting Started

### Prerequisites

- **Node.js** and **npm** (or yarn/pnpm)
- **Python 3.11+**
- An API key for an LLM provider:
  - **Anthropic-compatible** (e.g., Volcengine Ark) — set `ANTHROPIC_API_KEY` and optionally `ANTHROPIC_BASE_URL`
  - **OpenAI-compatible** — set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`

### 1. Configure Environment Variables

```bash
cd backend
cp .env.example .env
```

Edit `backend/.env` with your API keys. Example for an Anthropic-compatible provider:

```env
ANTHROPIC_API_KEY="your-api-key"
ANTHROPIC_BASE_URL="https://ark.cn-beijing.volces.com/api/plan"
```

Example for an OpenAI-compatible provider:

```env
OPENAI_API_KEY="your-api-key"
OPENAI_BASE_URL="https://api.openai.com/v1"
```

### 2. Install Dependencies

**Backend:**

```bash
cd backend
pip install -e .
```

> **Note:** `langgraph dev` requires Python 3.11+. If your system default is lower, create a conda environment first:
> ```bash
> conda create -n langgraph_env python=3.11 -y
> conda activate langgraph_env
> pip install -e .
> ```

**Frontend:**

```bash
cd frontend
npm install
```

### 3. Run Development Servers

**Both (recommended):**

```bash
make dev
```

This starts the backend (`langgraph dev`) and frontend (`vite`) concurrently.

- **Frontend:** http://localhost:5173/app
- **LangGraph Studio:** http://localhost:2024

**Alternatively, run separately:**

```bash
# Terminal 1 — Backend
cd backend && langgraph dev

# Terminal 2 — Frontend
cd frontend && npm run dev
```

> The frontend Vite dev server proxies `/api` requests to `http://127.0.0.1:8000`.

## Architecture Overview

### Frontend (React 19 + Vite 6 + Tailwind CSS 4)

| Component | Description |
|-----------|-------------|
| `App.tsx` | Thread manager: loads/saves thread list via localStorage, mounts `ChatSession` with `key={threadId}` for clean switching. |
| `Sidebar.tsx` | Collapsible left sidebar showing all historical threads. Supports select, delete, and new-chat actions. |
| `WelcomeScreen.tsx` | Clean landing page with search input. |
| `InputForm.tsx` | Textarea with Ctrl/Cmd+Enter submit, effort selector (Low/Medium/High), and model selector. |
| `ChatMessagesView.tsx` | Renders conversation history with ReactMarkdown (GFM tables supported), copy-to-clipboard, and source citation panels. |
| `ActivityTimeline.tsx` | Collapsible real-time timeline showing research steps with contextual icons. |

**Key frontend features:**
- **Multi-thread sidebar** — Each conversation is an independent thread persisted in `localStorage`. Switch freely without losing context.
- Real-time streaming via `@langchain/langgraph-sdk/react` `useStream` hook
- Automatic scroll-to-bottom on new messages
- Source panels per message: **知识库来源** (knowledge base) and **网络来源** (web sources)
- Markdown rendering with syntax-highlighted code blocks, **tables**, and blockquotes
- **PDF Export** — Download any AI answer as a print-ready PDF (light-theme, A4 layout) via `html-to-image` + `jsPDF`
- Graceful error display with retry button

### Backend (LangGraph + FastAPI)

The backend is a stateful LangGraph agent compiled into a research workflow:

| Node | Description |
|------|-------------|
| `generate_query` | Analyzes the user's question and generates diverse search queries via LLM. Supports native tool-calling when enabled. |
| `web_research` | Executes DuckDuckGo search (with SearXNG fallback) and synthesizes results into summaries with citations. |
| `rag_retrieve` | Retrieves top-k relevant document chunks from the local Chroma knowledge base. |
| `reflection` | Analyzes web summaries and knowledge base documents to identify gaps; generates follow-up queries if needed. Supports native tool-calling when enabled. |
| `finalize_answer` | Synthesizes all gathered information into a coherent, cited answer. |
| `agent_with_tools` *(reserved)* | ReAct-style agent node that lets the LLM invoke MCP or bound tools (not wired by default). |
| `execute_tool_calls` *(reserved)* | Executes tool calls produced by `agent_with_tools` and returns `ToolMessage` responses (not wired by default). |

### Agent Workflow

<img src="./agent.png" title="Agent Flow" alt="Agent Flow" width="100%">

1. **Generate Queries:** Based on user input, the LLM creates optimized search queries.
2. **Parallel Research:** Spawns parallel `web_research` nodes (one per query) **and** a `rag_retrieve` node simultaneously.
3. **Web Research:** For each query, performs web search and uses LLM to synthesize results into summaries.
4. **RAG Retrieval:** Searches the local Chroma vector store for relevant document chunks.
5. **Reflection:** Analyzes both web research and knowledge base results. If gaps exist, generates follow-up queries.
6. **Iterative Refinement:** Repeats web research and reflection with follow-up queries (up to the configured max loops).
7. **Finalize:** Combines all sources into a final answer with inline citations and source distinction.

### Memory & Thread Management

Thread metadata and long-term memory are now stored in **PostgreSQL** instead of browser `localStorage`:

- **Thread Metadata** — stored in the `thread_metadata` table (`thread_id`, `title`, `created_at`, `updated_at`).
- **Session Snapshots** — a lightweight JSON snapshot of each thread's final state is saved to `session_state`.
- **Long-Term Memory** — cross-session key-value pairs are stored in `agent_memory` (namespace → key → JSONB value).
- **LangGraph Checkpointer** — the graph compiles with `PostgresSaver` when the DB is reachable, enabling full thread-state checkpointing. If the DB is unavailable, it gracefully falls back to `MemorySaver`.

The frontend fetches the thread list from `/api/threads` on mount and synchronizes create/update/delete operations via REST. The only remaining `localStorage` usage is `lg-active-thread`, which remembers the last selected thread ID across page reloads.

- Clicking **New Chat** creates a fresh thread.
- Clicking a sidebar item switches threads without remounting the chat session (streams continue in the background).
- Deleting a thread removes both its metadata and session snapshot from PostgreSQL.

#### Automatic Memory Compression

To prevent long-term memory from growing unbounded, the system monitors the `research_topics` namespace after each answer:

- **Threshold-driven** — When the number of entries exceeds `memory_compression_threshold` (default 10), the oldest `memory_compression_batch_size` entries are fetched and passed to the LLM for summarization.
- **Compressed snapshot** — The LLM produces a concise summary, which is stored under a `_compressed_` key. The original raw entries are then deleted in a single batch operation.
- **Non-critical** — If compression fails (e.g., LLM timeout), the error is logged and the main research flow continues unaffected.

> **Development Note:** `langgraph dev` still uses an in-memory runtime by default. To enable PostgreSQL persistence during local development, set `POSTGRES_URI` and install the new dependencies (`pip install -e .`).

### LLM Provider Logic

The backend automatically selects the LLM client based on environment variables:

- **OpenAI-compatible API** is used when both `OPENAI_API_KEY` and `OPENAI_BASE_URL` are set.
- **Anthropic / Ark fallback** is used otherwise, reading `ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL`.

This dual-provider support lets you switch between OpenAI-compatible endpoints and Anthropic-compatible endpoints (e.g., Volcengine Ark) without code changes.

## RAG Knowledge Base

The built-in **Retrieval-Augmented Generation (RAG)** system allows the agent to query your local documents alongside web research.

### How It Works

- **Vector Store:** [Chroma](https://www.trychroma.com/) with persistence.
- **Embedding Model:** `BAAI/bge-small-zh-v1.5` (ONNX-based, optimized for Chinese semantic retrieval). Falls back to Chroma's default `all-MiniLM-L6-v2` if loading fails.
- **Hybrid Search** *(new)* — Combines BM25 keyword search with vector semantic search, then fuses results via Reciprocal Rank Fusion (RRF). A cross-encoder reranker further boosts precision by reordering candidates by relevance. All components gracefully degrade: if BM25 or reranking fails, the system falls back to pure vector search.
- **Document Loading:** Automatically loads and indexes documents from `backend/data/docs/` on first startup.
- **Supported Formats:** PDF (via PyPDFLoader), TXT, Markdown (`.md`, `.markdown`).
- **Text Splitting:** Recursive character splitter with 1000-character chunks and 200-character overlap.
- **Graceful Degradation:** If no documents are found, RAG is disabled, or retrieval fails, the system falls back to web-only research without errors.

### Adding Your Documents

1. Place your documents in `backend/data/docs/` (create the directory if it doesn't exist).
2. The vector store will be built automatically on the first request.
3. To rebuild the index after adding new documents, delete the `backend/data/chroma/` directory and restart the server.

### RAG in Answers

When knowledge base documents are retrieved and used:

- Paragraphs primarily based on the knowledge base are prefixed with `**【基于知识库】**`.
- Paragraphs primarily based on web research are prefixed with `**【基于网络搜索】**`.
- Inline citations use the format `[📄 source: filename.pdf]`.
- If no relevant documents are found, the answer explicitly states: `⚠️ 未在知识库中找到相关文档，以下回答完全基于网络搜索。`
- If web search returns no useful results, the answer states: `⚠️ 网络搜索未返回有效结果，以下回答完全基于知识库。`

## Tool-Calling & MCP (Reserved Capabilities)

The backend includes **optional, backward-compatible** support for two advanced LLM interaction patterns. Both are **disabled by default** so the existing research workflow is unaffected.

### Native Tool-Calling / Structured Output

By default, `generate_query` and `reflection` prompt the LLM to emit raw JSON and parse it manually with regex + Pydantic. When `tool_calling_enabled` is set to `true`, the agent switches to the LLM's native structured-output mechanism (`with_structured_output`), which uses the model's built-in tool-calling or JSON-mode capabilities for more reliable schema adherence.

```bash
export TOOL_CALLING_ENABLED=true
```

If the model does not support structured output, the system automatically falls back to the original manual JSON parsing.

### MCP (Model Context Protocol)

MCP allows the agent to discover and invoke tools from external MCP servers over stdio (or SSE) transport. This is useful for giving the agent access to filesystems, databases, APIs, command-line utilities, and more — without hard-coding tool logic into the agent itself.

**1. Install optional MCP dependencies:**

```bash
cd backend
pip install -e ".[mcp]"
```

**2. Configure MCP servers via environment variable:**

```bash
export MCP_ENABLED=true
export MCP_SERVERS='[
  {
    "name": "filesystem",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
  }
]'
```

**3. Use in custom graphs:**

The `agent_with_tools` and `execute_tool_calls` nodes are registered in the graph builder but **not wired into the default research pipeline**. You can wire them into a custom subgraph or replace the entry point to build a ReAct-style agent loop:

```
START → agent_with_tools → [has tool_calls?] → execute_tool_calls → agent_with_tools
                                      ↓ no
                              finalize_answer → END
```

> **Security Note:** MCP servers run as external processes. Only connect to trusted servers, and run them inside sandboxes (e.g., Docker) if they expose sensitive operations such as filesystem writes or shell command execution.

## Configuration

Agent behavior can be customized via environment variables or runtime config:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | API key for Anthropic-compatible LLM provider |
| `ANTHROPIC_BASE_URL` | — | Base URL for Anthropic-compatible API |
| `OPENAI_API_KEY` | — | API key for OpenAI-compatible provider |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Base URL for OpenAI-compatible API |
| `POSTGRES_URI` | `postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable` | PostgreSQL connection URI for thread metadata, memory, and LangGraph checkpointing |
| `query_generator_model` | `ark-code-latest` | Model for search query generation |
| `reflection_model` | `ark-code-latest` | Model for reflection and gap analysis |
| `answer_model` | `ark-code-latest` | Model for final answer synthesis |
| `number_of_initial_queries` | `3` | Number of initial search queries to generate |
| `max_research_loops` | `2` | Maximum research reflection loops |
| `rag_enabled` | `true` | Enable/disable RAG knowledge base retrieval |
| `rag_top_k` | `5` | Number of top documents to retrieve from knowledge base |
| `docs_dir` | `data/docs` | Directory containing documents to index for RAG |
| `chroma_persist_dir` | `data/chroma` | Directory to persist the Chroma vector store |
| `tool_calling_enabled` | `false` | Use native LLM tool-calling / structured-output instead of manual JSON parsing |
| `mcp_enabled` | `false` | Enable MCP (Model Context Protocol) tool server integration |
| `mcp_servers` | `""` | JSON-encoded list of MCP server configurations |
| `memory_enabled` | `true` | Enable cross-session long-term memory storage |
| `memory_compression_threshold` | `10` | Number of memory entries in a namespace before automatic compression triggers |
| `memory_compression_batch_size` | `10` | Number of oldest entries to compress in one batch |
| `hybrid_search_enabled` | `true` | Enable hybrid search (BM25 + vector) for RAG retrieval |
| `bm25_enabled` | `true` | Enable BM25 keyword search in hybrid retrieval |
| `rerank_enabled` | `true` | Enable cross-encoder reranking after hybrid retrieval |
| `hybrid_search_top_k` | `10` | Initial top-k candidates to retrieve from each search modality before fusion |
| `rerank_top_k` | `5` | Final top-k documents to return after reranking |

### Research Effort Levels

| Level | Initial Queries | Max Loops |
|-------|-----------------|-----------|
| Low | 1 | 1 |
| Medium | 3 | 3 |
| High | 5 | 10 |

## CLI Example

For quick one-off questions, run the agent directly from the command line without starting the web UI:

```bash
cd backend
python examples/cli_research.py "What are the latest trends in renewable energy?" \
  --initial-queries 3 \
  --max-loops 2 \
  --reasoning-model gemini-2.5-pro-preview-05-06
```

## Deployment

### Docker Compose (Production)

The included `docker-compose.yml` spins up Redis (pub/sub for streaming), PostgreSQL (state persistence), and the LangGraph API server.

**1. Build the Docker image:**

```bash
docker build -t gemini-fullstack-langgraph -f Dockerfile .
```

**2. Run the stack:**

```bash
GEMINI_API_KEY=<your_key> LANGSMITH_API_KEY=<your_key> docker-compose up
```

> For `docker-compose.yml`, you need a LangSmith API key from [LangSmith](https://smith.langchain.com/settings).

**3. Access the application:**

- **App:** http://localhost:8123/app/
- **API:** http://localhost:8123

### Notes on Production URLs

- The `Dockerfile` copies the built frontend into `/deps/frontend/dist` and mounts it at `/app` via FastAPI.
- The frontend `App.tsx` uses `http://localhost:8123` as the API URL in production builds and `http://localhost:2024` in development.
- Update the `apiUrl` in `frontend/src/App.tsx` if you are hosting the backend on a different domain.

## Technologies Used

### Frontend
- [React 19](https://react.dev/) — UI library
- [Vite 6](https://vitejs.dev/) — Build tool and dev server
- [TypeScript 5.7](https://www.typescriptlang.org/) — Type safety
- [Tailwind CSS 4.1](https://tailwindcss.com/) — Utility-first CSS framework
- [Shadcn UI](https://ui.shadcn.com/) — Re-usable component primitives (Radix UI)
- [Lucide React](https://lucide.dev/) — Icons
- [@langchain/langgraph-sdk](https://github.com/langchain-ai/langgraph) — SDK for streaming LangGraph agent events
- [React Markdown](https://github.com/remarkjs/react-markdown) + [remark-gfm](https://github.com/remarkjs/remark-gfm) — Markdown rendering with GFM tables

### Backend
- [LangGraph](https://github.com/langchain-ai/langgraph) — Framework for building stateful agent workflows
- [FastAPI](https://fastapi.tiangolo.com/) — Web framework for serving the frontend
- [LangChain](https://python.langchain.com/) / [LangChain Anthropic](https://python.langchain.com/) — LLM integrations
- [Chroma](https://www.trychroma.com/) — Open-source embedding database for RAG retrieval
- [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search) — Web search API with SearXNG fallback
- [PyPDF](https://github.com/py-pdf/pypdf) — PDF document loading
- [Pydantic](https://docs.pydantic.dev/) — Data validation and configuration

### Infrastructure
- [Redis](https://redis.io/) — Pub/sub broker for real-time streaming
- [PostgreSQL](https://www.postgresql.org/) — Persistence for threads, runs, and state

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
