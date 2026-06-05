# MY-AI

![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**A production-style vector database and RAG system built from scratch in C++**

| | |
|---|---|
| **Author** | Pratham Raj |
| **Institution** | NIT Patna |
| **Stack** | C++17 · cpp-httplib · Ollama · HNSW · KD-Tree · PCA |

---

## Overview

MY-AI is a full-stack vector search engine with a browser UI. It implements three nearest-neighbor algorithms (HNSW, KD-Tree, Brute Force), three distance metrics, real Ollama embeddings, and a complete Retrieval-Augmented Generation pipeline — all running locally.

```
Browser UI  →  REST API  →  Vector Engine  →  Ollama  →  LLM Answer
```

### Key Features

- **HNSW** — approximate O(log N) search (same family as Pinecone, Weaviate, Milvus)
- **KD-Tree** — exact axis-aligned partitioning for low dimensions
- **Brute Force** — exact baseline for correctness and benchmarking
- **PCA Scatter Plot** — live 2D projection of 16D semantic space
- **Document RAG** — `nomic-embed-text` (768D) + `llama3.2` generation
- **REST API** — full CRUD, benchmark, HNSW graph introspection

---

## Quick Start

### Prerequisites

| Tool | Purpose |
|---|---|
| MSYS2 / MinGW `g++` | C++17 compiler |
| Ollama | Local embeddings + LLM |
| Git (optional) | Version control |

### Install Ollama Models

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2
```

### Build & Run

```powershell
cd MY-AI
.\build.bat
.\db
```

Open **http://localhost:8080**

---

## Project Structure

```
MY-AI/
├── include/               # Modular headers
│   ├── config.hpp         # Constants and project metadata
│   ├── types.hpp          # Core data structures
│   ├── distances.hpp      # Cosine, Euclidean, Manhattan
│   ├── brute_force.hpp    # O(N) exact search
│   ├── kd_tree.hpp        # KD-Tree index
│   ├── hnsw.hpp           # HNSW graph index
│   ├── vector_db.hpp      # 16D demo vector store
│   ├── document_db.hpp    # 768D document index
│   ├── ollama_client.hpp  # Ollama HTTP client
│   ├── json_utils.hpp     # JSON helpers
│   ├── chunker.hpp        # Text chunking for RAG
│   ├── demo_data.hpp      # 20 pre-loaded demo vectors
│   └── api_server.hpp     # Route declarations
├── src/
│   ├── main.cpp           # Entry point
│   └── api_server.cpp     # REST route handlers
├── index.html             # Frontend (PCA, search, RAG UI)
├── httplib.h              # cpp-httplib (MIT)
├── build.bat              # Windows build script
├── README.md
└── PROJECT_MASTER_DOCUMENTATION.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/status` | System + Ollama status |
| `GET` | `/search` | K-NN search on demo vectors |
| `POST` | `/insert` | Insert demo vector |
| `DELETE` | `/delete/:id` | Delete demo vector |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure |
| `POST` | `/doc/insert` | Embed and store document |
| `GET` | `/doc/list` | List document chunks |
| `DELETE` | `/doc/delete/:id` | Delete document chunk |
| `POST` | `/doc/search` | Semantic retrieval |
| `POST` | `/doc/ask` | Full RAG pipeline |

See [PROJECT_MASTER_DOCUMENTATION.md](PROJECT_MASTER_DOCUMENTATION.md) for full API reference.

---

## Architecture Highlights

| Component | Complexity | Role |
|---|---|---|
| Brute Force | O(N·d) | Exact baseline |
| KD-Tree | O(log N) avg | Exact, low-D |
| HNSW | O(log N) approx | Production ANN |
| DocumentDB | HNSW @ 768D | Real embeddings |
| OllamaClient | HTTP | Embed + generate |

---

## Interview Use

This project demonstrates:

- Data structures (graphs, trees, heaps, hash maps)
- Algorithms (HNSW, KD-Tree, PCA, K-NN)
- Systems (REST API, threading, logging)
- AI/ML (embeddings, RAG, vector search)
- Engineering (modular C++, input validation, error handling)

Full interview prep (150+ Q&A, explanation scripts, viva notes) is in **PROJECT_MASTER_DOCUMENTATION.md**.

---

## License

MIT — Copyright (c) 2026 Pratham Raj, NIT Patna
