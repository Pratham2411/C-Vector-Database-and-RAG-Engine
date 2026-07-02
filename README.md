<div align="center">
  
# 🧠 MY-AI: C++ Vector Database & RAG Engine

**A highly-optimized, from-scratch Vector Search & Retrieval-Augmented Generation (RAG) engine.**

[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C?style=for-the-badge&logo=c%2B%2B)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-Build-064F8C?style=for-the-badge&logo=cmake)](https://cmake.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-white?style=for-the-badge&logo=ollama)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

*Built by [Pratham Raj](https://github.com/Pratham2411) — NIT Patna*

<br/>

<!-- 💡 TODO: Add your screen recording GIF here! -->
<!-- <img src="./assets/demo.gif" alt="MY-AI Demo" width="800"/> -->

</div>

---

## 📖 Project Overview

**MY-AI** is a portfolio-grade systems engineering project that demonstrates deep knowledge of data structures, algorithms, memory management, and modern AI pipelines. Instead of relying on a hosted cloud vector database (like Pinecone or Weaviate), **this project implements the core vector indexing algorithms completely from scratch in C++17.**

It features a dual-plane architecture:
1. **Vector Engine:** 16-dimensional semantic search utilizing custom **HNSW (Hierarchical Navigable Small World)** graphs, **KD-Trees**, and exact Brute Force K-NN search.
2. **Local RAG Pipeline:** Document chunking, local embeddings (`nomic-embed-text`), and generative Q&A (`llama3.2`) powered locally by Ollama.

Everything is exposed via a lightweight REST API and visualized through a modern, responsive PCA scatter-plot dashboard.

---

## ✨ Core Features

- **Built from Scratch:** No external vector libraries (no FAISS, no hnswlib). Implements HNSW graph traversal and KD-Tree spatial partitioning purely in standard C++.
- **Concurrent & Thread-Safe:** Uses `std::shared_mutex` (Readers-Writer locks) to allow highly concurrent read queries without blocking, while safely locking during document insertion/deletion.
- **Algorithm Benchmarking:** Live REST endpoints to benchmark the microsecond latency difference between `O(log N)` HNSW and `O(N)` Brute Force algorithms.
- **Transactional Rollbacks:** Robust error handling that rolls back partial document insertions if embedding generation fails mid-chunk.
- **Zero-Dependency UI:** A beautiful, responsive, single-file HTML/CSS/JS frontend dashboard utilizing CSS variables and canvas API for live PCA (Principal Component Analysis) projection.

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    subgraph Frontend["Browser UI"]
        Dashboard[PCA Dashboard]
        Chat[RAG Chat Interface]
    end

    subgraph Backend["C++ REST API Server"]
        API[HTTP Routes]
        
        subgraph Storage["Vector Stores (Readers-Writer Locks)"]
            DocDB[Document DB<br/>(768D)]
            VecDB[Demo Vector DB<br/>(16D)]
        end
        
        subgraph Indexes["Algorithms"]
            HNSW[HNSW Graph]
            KD[KD-Tree]
            BF[Brute Force]
        end
    end

    subgraph External["Local AI"]
        Ollama[Ollama<br/>llama3.2 + nomic]
    end

    Dashboard <-->|REST| API
    Chat <-->|REST| API
    API --> DocDB & VecDB
    DocDB --> HNSW & BF
    VecDB --> HNSW & KD & BF
    API <-->|HTTP| Ollama
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **C++17 Compiler** (MSVC, GCC, or Clang)
- **CMake** (3.16+)
- **Ollama** (Running locally on port `11434`)

### 2. Install AI Models
Pull the required embedding and generation models:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### 3. Build & Run (Docker - Recommended)
The easiest way to run the database without setting up a C++ toolchain:
```bash
docker build -t my-ai .
docker run -p 8080:8080 my-ai
```

### 4. Build & Run (Native Windows/Linux)
```bash
cmake -S . -B build
cmake --build build --config Release
./build/db
```

The REST API and UI will be available at `http://localhost:8080`.

---

## ⚡ Performance & Benchmarks

The custom implementation proves the mathematical efficiency of Approximate Nearest Neighbors (ANN). 

| Algorithm | Big-O Complexity | 10k Vectors Latency | Accuracy |
|-----------|------------------|---------------------|----------|
| **Brute Force** | $O(N \cdot d)$ | ~45,000 µs | 100% (Exact) |
| **KD-Tree** | $O(\log N)$ | ~2,100 µs | 100% (Exact for Euclidean) |
| **HNSW** | $O(\log N)$ | **~150 µs** | ~98% (Approximate) |

*(Note: Benchmarks vary by hardware. Run the `/benchmark` endpoint via the UI to see your local results).*

---

## 📂 Repository Structure

```text
MY-AI/
├── include/                 # Core logic and header-only implementations
│   ├── vector_db.hpp        # 16D Thread-safe Demo Store
│   ├── document_db.hpp      # 768D Thread-safe RAG Store
│   ├── hnsw.hpp             # Hierarchical Navigable Small World algorithm
│   ├── kd_tree.hpp          # K-Dimensional Tree algorithm
│   └── ...                  # Distance metrics, json utilities, chunkers
├── src/
│   ├── api_server.cpp       # cpp-httplib REST endpoint routing
│   └── main.cpp             # Entry point & server bootstrapping
├── index.html               # Vanilla JS/CSS Dashboard UI
├── Dockerfile               # Containerization for easy testing
├── .clang-format            # C++ Code styling rules
└── PROJECT_MASTER_DOCUMENTATION.md # 95KB Deep-dive technical documentation
```

---

## 📚 Detailed Documentation & Interview Prep

For an extremely deep dive into the math, architecture, and design decisions behind this project—including 150+ simulated technical interview questions—please read the [PROJECT MASTER DOCUMENTATION](PROJECT_MASTER_DOCUMENTATION.md).

---

## ⚖️ License

Distributed under the MIT License. Copyright (c) 2026 Pratham Raj.
