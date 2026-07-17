# MY-AI — Features & Demo Guide

Use this guide when showing the project to others (interviews, portfolio reviews, demos).  
Open the app at **http://localhost:8080** after starting the server (see `run.tex`).

---

## What This Project Does (30-second pitch)

> "This is a C++ vector database I built from scratch — no Pinecone, no FAISS. It implements HNSW, KD-Tree, and Brute Force search, exposes a REST API, and includes a full RAG pipeline with local Ollama models. Everything runs on your machine."

---

## Dashboard Layout

| Area | What it shows |
|------|----------------|
| **Left panel** | Search controls, algorithm picker, metrics, insert demo vectors, benchmark button |
| **Center** | Live 2D PCA scatter plot of all vectors (demo + documents) |
| **Right panel** | Three tabs: **SEARCH**, **DOCUMENTS**, **ASK AI** |
| **Header** | Project badges, Ollama online/offline status, vector count |

---

## Feature 1: Vector Search (Demo Store)

**What it is:** Semantic search over ~10,000 pre-loaded 16-dimensional demo vectors (CS, math, food, sports topics).

**How to demo:**
1. In the left panel, type a query like `binary tree`, `sushi`, or `basketball`.
2. Click **SEARCH**.
3. Point out:
   - Results appear in the **SEARCH** tab (right panel) with distance scores.
   - Matching dots highlight on the scatter plot; lines connect query to hits.
   - **Search Latency** shows microseconds for the chosen algorithm.

**Try different algorithms** (left panel):
- **HNSW** — fastest (~150 µs), approximate nearest neighbors.
- **KD-TREE** — exact search, good for low dimensions.
- **BRUTE** — scans every vector; slowest but 100% accurate baseline.

**Try different metrics:** Cosine, Euclidean, Manhattan.

**Adjust Top-K** slider (1–10) to change how many results return.

---

## Feature 2: Algorithm Benchmark

**What it is:** Side-by-side speed comparison of all three algorithms on the same query.

**How to demo:**
1. Enter any search query (or use the last one).
2. Click **COMPARE ALL ALGOS** in the left panel.
3. Show the bar chart in the SEARCH tab — HNSW is usually dramatically faster than brute force.

**Talking point:** "This proves why production vector DBs use graph indexes instead of linear scan."

---

## Feature 3: HNSW Graph Visualization

**What it is:** Live view of the Hierarchical Navigable Small World graph structure.

**How to demo:**
1. Scroll down in the **SEARCH** tab to **HNSW Graph Layers**.
2. Explain each layer: top layers = long jumps, bottom layer = fine-grained links.
3. Optional API: `GET http://localhost:8080/hnsw-info` returns full graph JSON.

---

## Feature 4: Insert & Delete Demo Vectors

**What it is:** Add your own labeled vectors to the demo store (client-side 16D embedding from keywords).

**How to demo:**
1. In **Insert Demo Vector**, type a description (e.g. `neural networks`).
2. Pick a category (CS, Math, Food, Sports).
3. Click **+ INSERT** — a new dot appears on the scatter plot.
4. Search for your term to show it ranks highly.
5. Delete a result with the **×** button on any result card.

**Talking point:** "Inserts update the HNSW graph and KD-Tree in real time with thread-safe locks."

---

## Feature 5: PCA Scatter Plot

**What it is:** 16D vectors projected to 2D using Principal Component Analysis so you can *see* semantic clusters.

**How to demo:**
1. Hover over dots — tooltip shows category and metadata.
2. After a search, watch the white star (query) and highlighted neighbors.
3. Green dots = inserted RAG documents (after you add docs in Feature 6).

---

## Feature 6: Document Store & RAG Pipeline

**What it is:** Real document ingestion with Ollama embeddings (768D `nomic-embed-text`), chunking, and vector search.

**Requires:** Ollama running with `nomic-embed-text` and `llama3.2` pulled.

**How to demo:**
1. Switch to the **DOCUMENTS** tab.
2. Check **Ollama Status** — should show ONLINE with model names.
3. Paste sample text, for example:

   **Title:** `Dynamic Programming`  
   **Text:** A short paragraph about memoization and optimal substructure.

4. Click **EMBED & INSERT** — text is chunked, embedded, and stored.
5. Show the new entry in **Stored Documents** list.
6. Long text auto-splits into overlapping chunks (250 words, 30 overlap).

**Talking point:** "If embedding fails mid-insert, the server rolls back partial chunks — no half-written data."

---

## Feature 7: Ask AI (RAG Chat)

**What it is:** Retrieval-Augmented Generation — finds relevant document chunks, then asks the local LLM to answer.

**How to demo:**
1. Insert at least one document (Feature 6).
2. Switch to **ASK AI** tab.
3. Ask: `What is dynamic programming?` or `Summarize what I inserted.`
4. Click **ASK AI**.
5. Show:
   - The generated answer (from `llama3.2`).
   - Context chips below — click to expand source chunks used.
   - Top-K selector (2, 3, or 5 chunks as context).

**Talking point:** "Search is pure C++ vector math; generation is delegated to Ollama locally — no cloud API keys."

---

## REST API Reference (for technical demos)

All endpoints run on `http://localhost:8080`.

### Demo vector store (16D)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/items` | List all demo vectors |
| GET | `/search?v=...&k=5&algo=hnsw&metric=cosine` | Vector search |
| POST | `/insert` | Add a vector (`metadata`, `category`, `embedding`) |
| DELETE | `/delete/{id}` | Remove a vector |
| GET | `/benchmark?v=...&k=5&metric=cosine` | Compare algorithm speeds |
| GET | `/hnsw-info` | HNSW graph structure |
| GET | `/stats` | Store size, dims, supported algos/metrics |

### Document / RAG store (768D)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/doc/insert` | Insert document (`title`, `text`) |
| GET | `/doc/list` | List stored document chunks |
| DELETE | `/doc/delete/{id}` | Delete a chunk |
| POST | `/doc/search` | Semantic search (`question`, `k`) |
| POST | `/doc/ask` | Full RAG Q&A (`question`, `k`) |

### System

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/status` | Project info, Ollama status, counts |
| GET | `/` | Web dashboard (`index.html`) |

### Example curl commands

```bash
# Health check
curl http://localhost:8080/status

# List demo vectors
curl http://localhost:8080/items

# Ask RAG a question
curl -X POST http://localhost:8080/doc/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What is HNSW?\",\"k\":3}"
```

---

## Suggested 5-Minute Demo Script

1. **Start** — Open dashboard, mention it's pure C++17, no vector libraries.
2. **Search** — Query `sushi`, show HNSW results and latency.
3. **Benchmark** — Run compare, highlight speed gap vs brute force.
4. **Plot** — Hover clusters, show PCA projection.
5. **RAG** — Insert a short note, ask a question in ASK AI tab.
6. **Close** — Mention thread safety, rollback on failure, and REST API.

---

## What Works Without Ollama

| Feature | Works offline? |
|---------|----------------|
| Vector search (demo store) | Yes |
| Benchmark | Yes |
| HNSW visualization | Yes |
| Insert demo vectors | Yes |
| Insert documents | No |
| Ask AI (RAG) | No |

Vector search is the core demo; RAG is the bonus when Ollama is available.

---

## Files Worth Mentioning in a Technical Review

| File | Role |
|------|------|
| `include/hnsw.hpp` | HNSW graph implementation |
| `include/kd_tree.hpp` | KD-Tree implementation |
| `include/vector_db.hpp` | Thread-safe demo vector store |
| `include/document_db.hpp` | 768D RAG document store |
| `src/api_server.cpp` | All REST routes |
| `index.html` | Full dashboard UI (no framework) |
| `PROJECT_MASTER_DOCUMENTATION.md` | Deep technical write-up |

---

*Built by Pratham Raj — NIT Patna*
