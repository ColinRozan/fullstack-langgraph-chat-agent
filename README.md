# Fullstack LangGraph Quickstart

This project demonstrates a fullstack application using a React frontend and a LangGraph-powered backend agent. The agent is designed to perform comprehensive research on a user's query by dynamically generating search terms, querying the web using Google Search, reflecting on the results to identify knowledge gaps, and iteratively refining its search until it can provide a well-supported answer with citations. This application serves as an example of building research-augmented conversational AI using LangGraph and Anthropic-compatible models.

> **Note:** The models used in this project are based on the Anthropic protocol and can be switched as needed.

<img src="./app.png" title="Fullstack LangGraph" alt="Fullstack LangGraph" width="90%">

## Features

- 💬 Fullstack application with a React frontend and LangGraph backend.
- 🧠 Powered by a LangGraph agent for advanced research and conversational AI.
- 🔍 Dynamic search query generation using Anthropic-compatible models.
- 🌐 Integrated web research via DuckDuckGo with SearXNG fallback.
- 📚 **RAG Knowledge Base**: Local document retrieval powered by Chroma vector store, supporting PDF, TXT, and Markdown files.
- 🤔 Reflective reasoning to identify knowledge gaps and refine searches.
- 📄 Generates answers with citations from both web sources and knowledge base documents.
- 🎯 **Research Depth Control**: Choose between Low, Medium, and High effort modes to adjust query count and iteration loops.
- 🔄 Hot-reloading for both frontend and backend during development.
- 🎨 Modern UI with real-time activity timeline showing each research step.

## Project Structure

```
.
├── frontend/                     # React + Vite frontend application
│   ├── src/
│   │   ├── App.tsx              # Main application component with LangGraph SDK stream
│   │   ├── components/          # UI components (Chat, WelcomeScreen, ActivityTimeline)
│   │   └── ...
│   ├── package.json
│   └── vite.config.ts
├── backend/                      # LangGraph + FastAPI backend
│   ├── src/agent/
│   │   ├── graph.py             # Core LangGraph agent definition
│   │   ├── state.py             # TypedDict state definitions
│   │   ├── prompts.py           # LLM prompt templates
│   │   ├── configuration.py     # Agent configuration (models, RAG settings)
│   │   ├── knowledge_base.py    # RAG retrieval with Chroma vector store
│   │   ├── tools_and_schemas.py # Pydantic schemas for LLM outputs
│   │   └── utils.py             # Helper utilities
│   ├── examples/
│   │   └── cli_research.py      # CLI example for running research
│   └── data/
│       ├── docs/                # Place your documents here for RAG
│       └── chroma/              # Chroma vector store persistence
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## Getting Started: Development and Local Testing

Follow these steps to get the application running locally for development and testing.

**1. Prerequisites:**

-   Node.js and npm (or yarn/pnpm)
-   Python 3.11+
-   **`ANTHROPIC_API_KEY`**: The backend agent requires an Anthropic API key (or a compatible provider key).
    1.  Navigate to the `backend/` directory.
    2.  Create a file named `.env` by copying the `backend/.env.example` file.
    3.  Open the `.env` file and add your API key: `ANTHROPIC_API_KEY="YOUR_ACTUAL_API_KEY"`

    > The models used in this project are based on the Anthropic protocol and can be switched to any compatible provider by adjusting the `ANTHROPIC_BASE_URL` and model names in the `.env` file.

**2. Install Dependencies:**

**Backend:**

```bash
cd backend
pip install .
```

**Frontend:**

```bash
cd frontend
npm install
```

**3. Run Development Servers:**

**Backend & Frontend:**

```bash
make dev
```
This will run the backend and frontend development servers.    Open your browser and navigate to the frontend development server URL (e.g., `http://localhost:5173/app`).

_Alternatively, you can run the backend and frontend development servers separately. For the backend, open a terminal in the `backend/` directory and run `langgraph dev`. The backend API will be available at `http://127.0.0.1:2024`. It will also open a browser window to the LangGraph UI. For the frontend, open a terminal in the `frontend/` directory and run `npm run dev`. The frontend will be available at `http://localhost:5173`._

## Architecture Overview

### Frontend (React + Vite)

The frontend is a modern React application built with Vite, featuring:

- **Real-time Streaming**: Uses `@langchain/langgraph-sdk/react` `useStream` hook to stream agent events in real-time.
- **Activity Timeline**: Displays each research step live — Query Generation → Web Research → RAG Retrieval → Reflection → Finalizing Answer.
- **Research Depth Selector**: Users can choose `Low` (1 query, 1 loop), `Medium` (3 queries, 3 loops), or `High` (5 queries, 10 loops) effort modes.
- **Model Selection**: Supports switching between different LLM models for query generation, reflection, and answer synthesis.
- **Source Citations**: Visual distinction between web sources [🌐] and knowledge base sources [📄] in the final answer.
- **UI Stack**: React 18, Tailwind CSS, Shadcn UI components, Radix UI primitives.

### Backend (LangGraph + FastAPI)

The backend is built on LangGraph with a stateful agent graph:

| Node | Description |
|------|-------------|
| `generate_query` | Generates diverse search queries based on the user's question |
| `web_research` | Performs web search via DuckDuckGo (with SearXNG fallback) and synthesizes results |
| `rag_retrieve` | Retrieves relevant documents from the local Chroma knowledge base |
| `reflection` | Analyzes gathered information to identify knowledge gaps and decide whether to continue |
| `finalize_answer` | Synthesizes web research and knowledge base results into a coherent, cited answer |

### Agent Workflow

<img src="./agent.png" title="Agent Flow" alt="Agent Flow" width="50%">

1.  **Generate Initial Queries:** Based on your input, it generates a set of initial search queries using a configured LLM.
2.  **Parallel Research:** Spawns parallel `web_research` nodes (one per query) **and** a `rag_retrieve` node to query the local knowledge base simultaneously.
3.  **Web Research:** For each query, performs DuckDuckGo search (with SearXNG fallback) and uses LLM to synthesize search results into summaries with citations.
4.  **RAG Retrieval:** Searches the local Chroma vector store for the top-k most relevant document chunks related to the research topic.
5.  **Reflection & Knowledge Gap Analysis:** The agent analyzes both web research summaries and knowledge base documents to determine if the information is sufficient. If RAG documents are present, the reflection prompt explicitly considers both sources.
6.  **Iterative Refinement:** If gaps are found, it generates follow-up queries and repeats the web research and reflection steps (up to a configured maximum number of loops).
7.  **Finalize Answer:** Once research is sufficient, the agent synthesizes all gathered information into a coherent answer, with inline citations distinguishing web sources `[🌐 Title](URL)` from knowledge base sources `[📄 source: filename.pdf]`.

## RAG Knowledge Base

This project includes a built-in **Retrieval-Augmented Generation (RAG)** system that allows the agent to query your local documents alongside web research.

### How It Works

- **Vector Store**: Uses [Chroma](https://www.trychroma.com/) with the lightweight `all-MiniLM-L6-v2` ONNX embedding model (bundled with Chroma, no external API keys needed).
- **Document Loading**: Automatically loads and indexes documents from `backend/data/docs/` on first startup.
- **Supported Formats**: PDF, TXT, Markdown (`.md`, `.markdown`).
- **Text Splitting**: Documents are split into 1000-character chunks with 200-character overlap for optimal retrieval.
- **Graceful Degradation**: If no documents are found or RAG is disabled, the system falls back to web-only research without errors.

### Adding Your Documents

1. Place your documents in `backend/data/docs/` (create the directory if it doesn't exist).
2. The vector store will be built automatically on the first request.
3. To rebuild the index after adding new documents, delete the `backend/data/chroma/` directory and restart the server.

### RAG in Answers

When knowledge base documents are retrieved and used:

- Relevant paragraphs are prefixed with `**【基于知识库】**` (Based on Knowledge Base).
- Inline citations use the format `[📄 source: filename.pdf]`.
- If no relevant documents are found, the answer explicitly states: `⚠️ 未在知识库中找到相关文档，以下回答完全基于网络搜索。`

## Configuration

The agent behavior can be customized via environment variables (in `backend/.env`) or passed at runtime:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | API key for Anthropic-compatible LLM provider |
| `ANTHROPIC_BASE_URL` | — | Base URL for Anthropic-compatible API |
| `OPENAI_API_KEY` | — | Alternative: use OpenAI-compatible API |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Base URL for OpenAI-compatible API |
| `query_generator_model` | `kimi-k3` | Model for search query generation |
| `reflection_model` | `kimi-k3` | Model for reflection and gap analysis |
| `answer_model` | `kimi-k3` | Model for final answer synthesis |
| `number_of_initial_queries` | `3` | Number of initial search queries to generate |
| `max_research_loops` | `2` | Maximum research reflection loops |
| `rag_enabled` | `true` | Enable/disable RAG knowledge base retrieval |
| `rag_top_k` | `5` | Number of top documents to retrieve from knowledge base |

> **Note:** If both `OPENAI_API_KEY` and `OPENAI_BASE_URL` are set, the agent will use the OpenAI-compatible API. Otherwise, it falls back to Anthropic/Ark.

## CLI Example

For quick one-off questions you can execute the agent from the command line. The
script `backend/examples/cli_research.py` runs the LangGraph agent and prints the
final answer:

```bash
cd backend
python examples/cli_research.py "What are the latest trends in renewable energy?"
```


## Deployment

In production, the backend server serves the optimized static frontend build. LangGraph requires a Redis instance and a Postgres database. Redis is used as a pub-sub broker to enable streaming real time output from background runs. Postgres is used to store assistants, threads, runs, persist thread state and long term memory, and to manage the state of the background task queue with 'exactly once' semantics. For more details on how to deploy the backend server, take a look at the [LangGraph Documentation](https://langchain-ai.github.io/langgraph/concepts/deployment_options/). Below is an example of how to build a Docker image that includes the optimized frontend build and the backend server and run it via `docker-compose`.

_Note: For the docker-compose.yml example you need a LangSmith API key, you can get one from [LangSmith](https://smith.langchain.com/settings)._

_Note: If you are not running the docker-compose.yml example or exposing the backend server to the public internet, you should update the `apiUrl` in the `frontend/src/App.tsx` file to your host. Currently the `apiUrl` is set to `http://localhost:8123` for docker-compose or `http://localhost:2024` for development._

**1. Build the Docker Image:**

   Run the following command from the **project root directory**:
   ```bash
   docker build -t fullstack-langgraph -f Dockerfile .
   ```
**2. Run the Production Server:**

   ```bash
   ANTHROPIC_API_KEY=<your_anthropic_api_key> LANGSMITH_API_KEY=<your_langsmith_api_key> docker-compose up
   ```

Open your browser and navigate to `http://localhost:8123/app/` to see the application. The API will be available at `http://localhost:8123`.

## Technologies Used

### Frontend
- [React](https://reactjs.org/) (with [Vite](https://vitejs.dev/)) - Frontend framework and build tool.
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework.
- [Shadcn UI](https://ui.shadcn.com/) - Re-usable component primitives.
- [@langchain/langgraph-sdk](https://github.com/langchain-ai/langgraph) - SDK for streaming LangGraph agent events.

### Backend
- [LangGraph](https://github.com/langchain-ai/langgraph) - Framework for building stateful agent workflows.
- [FastAPI](https://fastapi.tiangolo.com/) - Modern, fast web framework for the API server.
- [LangChain Anthropic](https://python.langchain.com/) / [LangChain Community](https://python.langchain.com/) - LLM integrations and document loaders.
- [Chroma](https://www.trychroma.com/) - Open-source embedding database for RAG retrieval.
- [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search) - Web search API with fallback to SearXNG.

### Models
- [Anthropic Claude](https://www.anthropic.com/claude) (or any Anthropic-compatible model) - Default LLM for query generation, reflection, and answer synthesis.
- OpenAI-compatible models - Alternative LLM provider via `OPENAI_API_KEY` / `OPENAI_BASE_URL`.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details. 
