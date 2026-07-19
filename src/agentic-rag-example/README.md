# AgenticRAG Example (LangGraph + Tavily + Chroma + LLM)

This is a separate example from the existing chatbot.

## What it does

- Uses a LangGraph router node to classify each query into one route:
  - `web_search` via Tavily for current/live info
  - `chroma_rag` for local docs retrieval from Chroma Vector DB
  - `general_llm` for generic reasoning/chat
- Uses a final `synthesize_response` node to generate the final answer.
- Returns workflow trace logs and source snippets to the UI.

## Workflow Diagram

```mermaid
flowchart TD
  A[User Query] --> B[classify_query]

  B -->|web_search| C[web_search node\nTavily API]
  B -->|chroma_rag| D[chroma_rag node\nChroma retrieval]
  B -->|general_llm| E[general_llm node]

  C --> F[synthesize_response]
  D --> F
  E --> F

  F --> G[Final AI Answer]

  subgraph Observability
    H[Workflow Trace\nnode, status, duration]
    I[Sources\nweb links or local snippets]
  end

  C -. emits .-> H
  D -. emits .-> H
  E -. emits .-> H
  F -. emits .-> H

  C -. emits .-> I
  D -. emits .-> I
```

## Run

1. Ensure `.env` contains:

```bash
GROQ_API_KEY=...
TAVILY_API_KEY=...
# Optional
LLM_PROVIDER=groq
CHROMA_URL=http://localhost:8000
CHROMA_COLLECTION=agentic_rag_docs
EMBEDDING_MODEL=Xenova/bge-m3
PORT=3100
```

2. Start Chroma server (separate terminal):

```bash
npm run chroma
```

3. Start this example:

```bash
npm run agentic:rag:web
```

4. Open:

```bash
http://localhost:3100/agentic-rag
```

## Notes

- Chroma indexing is lazy and runs on first query.
- Only `.txt` and `.md` files under `docs/` are indexed in this example.
- Workflow details are returned by `POST /api/chat`.
