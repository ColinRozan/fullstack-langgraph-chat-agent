# Stage 1: Build React Frontend
FROM node:20-alpine AS frontend-builder

# Set working directory for frontend
WORKDIR /app/frontend

# Copy frontend package files and install dependencies
COPY frontend/package.json ./
COPY frontend/package-lock.json ./
# If you use yarn or pnpm, adjust accordingly (e.g., copy yarn.lock or pnpm-lock.yaml and use yarn install or pnpm install)
RUN npm install

# Copy the rest of the frontend source code
COPY frontend/ ./

# Build the frontend
RUN npm run build

# Stage 2: Python Backend
FROM docker.io/langchain/langgraph-api:3.11

# Patch missing feature-flag required by the bundled langgraph_runtime_postgres
RUN echo "PREFER_GRPC_CHECKPOINTER = False" >> /usr/local/lib/python3.11/site-packages/langgraph_api/feature_flags.py

# -- Copy built frontend from builder stage --
# The app.py expects the frontend build to be at ../frontend/dist relative to its own location.
# If app.py is at /deps/backend/src/agent/app.py, then ../frontend/dist resolves to /deps/frontend/dist.
COPY --from=frontend-builder /app/frontend/dist /deps/frontend/dist
# -- End of copying built frontend --

# -- Adding local package . --
ADD backend/ /deps/backend
# -- End of local package . --

ENV PYTHONPATH="/deps/backend/src:${PYTHONPATH}"
ENV LANGGRAPH_HTTP='{"app": "/deps/backend/src/agent/app.py:app"}'
ENV LANGSERVE_GRAPHS='{"agent": "/deps/backend/src/agent/graph.py:graph"}'
ENV PIP_NO_PARALLEL=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_PROGRESS_BAR=off

# Install backend dependencies (quiet to avoid rich progress threads)
RUN cd /deps/backend && \
    pip install --no-cache-dir --quiet setuptools wheel && \
    pip install --no-cache-dir --quiet --no-build-isolation -e .

WORKDIR /deps/backend
