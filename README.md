# MY-AI

![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**A local vector database and Retrieval-Augmented Generation system built from scratch in C++17.**

MY-AI is a portfolio-grade systems project that demonstrates vector search internals, REST API design, browser visualization, and local LLM integration without relying on a hosted AI service.

| Field | Details |
|---|---|
| Author | Pratham Raj |
| Institution | NIT Patna |
| Core stack | C++17, cpp-httplib, Ollama, HNSW, KD-Tree, PCA |
| Runtime | Local Windows app, served at `http://localhost:8080` |

---

## What It Does

MY-AI has two connected workflows:

1. **Vector search demo:** stores 16-dimensional semantic demo vectors and lets you compare HNSW, KD-Tree, and brute-force K-NN search.
2. **Document RAG:** chunks documents, embeds them with Ollama's `nomic-embed-text`, retrieves relevant chunks, and asks `llama3.2` to answer with local context.

```text
Browser UI -> REST API -> Vector Engine -> Ollama -> Local LLM Answer
```

---

## Highlights

- **HNSW index from scratch** for approximate nearest-neighbor search.
- **KD-Tree and brute-force baselines** for comparison and correctness checks.
- **Three distance metrics:** cosine, Euclidean, and Manhattan.
- **Live PCA scatter plot** that projects 16D demo vectors into 2D.
- **Local RAG pipeline** with document chunking, embeddings, retrieval, and generation.
- **REST API** for search, insert/delete, benchmark, status, graph introspection, and document Q&A.
- **Thread-safe in-memory stores** with input validation and rollback on failed document inserts.

---

## Quick Start

### Prerequisites

| Tool | Purpose |
|---|---|
| MSYS2 / MinGW `g++` | C++17 compiler |
| Ollama | Local embeddings and LLM generation |
| Git | Version control |

### Install Ollama Models

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

### Build And Run

```powershell
cd MY-AI
.\build.bat
.\db.exe
```

Open:

```text
http://localhost:8080
```

The vector search demo works immediately. RAG features require Ollama to be running with the models above installed.

### Optional CMake Build

```powershell
cmake -S . -B build
cmake --build build
```

---

## Project Structure

```text
MY-AI/
|-- include/                 # Modular C++ headers
|   |-- config.hpp           # Constants and project metadata
|   |-- types.hpp            # Core data structures
|   |-- distances.hpp        # Cosine, Euclidean, Manhattan
|   |-- brute_force.hpp      # Exact O(N) search baseline
|   |-- kd_tree.hpp          # KD-Tree index
|   |-- hnsw.hpp             # HNSW graph index
|   |-- vector_db.hpp        # 16D demo vector store
|   |-- document_db.hpp      # RAG document vector index
|   |-- ollama_client.hpp    # Ollama HTTP client
|   |-- json_utils.hpp       # Lightweight JSON helpers
|   |-- chunker.hpp          # Text chunking for RAG
|   |-- demo_data.hpp        # Preloaded demo vectors
|   `-- api_server.hpp       # Route declarations
|-- src/
|   |-- main.cpp             # Entry point
|   `-- api_server.cpp       # REST route handlers
|-- index.html               # Browser UI
|-- httplib.h                # cpp-httplib dependency
|-- build.bat                # Windows build script
|-- PROJECT_MASTER_DOCUMENTATION.md
`-- README.md
```

---

## API Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the web UI |
| `GET` | `/status` | Project, Ollama, and document status |
| `GET` | `/stats` | Demo vector database stats |
| `GET` | `/items` | List demo vectors |
| `GET` | `/search` | K-NN search over demo vectors |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete a demo vector |
| `GET` | `/benchmark` | Compare brute force, KD-Tree, and HNSW |
| `GET` | `/hnsw-info` | Inspect HNSW graph layers and edges |
| `POST` | `/doc/insert` | Chunk, embed, and store a document |
| `GET` | `/doc/list` | List document chunks |
| `DELETE` | `/doc/delete/:id` | Delete a document chunk |
| `POST` | `/doc/search` | Retrieve relevant document chunks |
| `POST` | `/doc/ask` | Run the full RAG answer pipeline |

Example:

```powershell
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05&k=5&metric=cosine&algo=hnsw"
```

---

## Architecture

| Component | Complexity / Role |
|---|---|
| Brute Force | Exact baseline, `O(N*d)` |
| KD-Tree | Exact low-dimensional search for Euclidean distance |
| HNSW | Approximate graph-based ANN search |
| VectorDB | 16D teaching/demo vector database |
| DocumentDB | 768D document embeddings for RAG |
| OllamaClient | Local embedding and generation calls |
| Frontend | PCA visualization, benchmark UI, document RAG workflow |

---

## Why This Project Is Valuable

This project shows practical understanding of:

- Data structures: graphs, trees, heaps, hash maps
- Algorithms: HNSW, KD-Tree, K-NN, PCA, distance metrics
- Systems: C++ REST server, threading, logging, validation
- AI engineering: embeddings, vector retrieval, RAG prompting
- Product polish: browser UI, latency reporting, graph visualization

For the detailed architecture, full API reference, viva notes, and interview preparation, see [PROJECT_MASTER_DOCUMENTATION.md](PROJECT_MASTER_DOCUMENTATION.md).

---

## License

MIT License. Copyright (c) 2026 Pratham Raj, NIT Patna.
