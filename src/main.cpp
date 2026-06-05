#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#endif

#include "api_server.hpp"
#include "config.hpp"
#include "demo_data.hpp"
#include "document_db.hpp"
#include "logging.hpp"
#include "ollama_client.hpp"
#include "vector_db.hpp"

#include <iostream>

int main() {
    myai::VectorDB db(myai::DEMO_DIMS);
    myai::DocumentDB docDB;
    myai::OllamaClient ollama;

    myai::loadDemo(db);

    const bool ollamaUp = ollama.isAvailable();
    std::cout << "=== " << myai::PROJECT_NAME << " Engine ===" << std::endl;
    std::cout << "Author: " << myai::AUTHOR_NAME << " - " << myai::AUTHOR_ORG << std::endl;
    std::cout << "http://localhost:" << myai::DEFAULT_PORT << std::endl;
    std::cout << db.size() << " demo vectors | " << myai::DEMO_DIMS
              << " dims | HNSW+KD-Tree+BruteForce" << std::endl;
    std::cout << "Ollama: " << (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)") << std::endl;
    if (ollamaUp) {
        std::cout << "  embed model: " << ollama.embedModel
                  << "  gen model: " << ollama.genModel << std::endl;
    }

    httplib::Server svr;
    myai::registerRoutes(svr, db, docDB, ollama);

    myai::log(myai::LogLevel::Info, "Listening on 0.0.0.0:" + std::to_string(myai::DEFAULT_PORT));
    if (!svr.listen("0.0.0.0", myai::DEFAULT_PORT)) {
        myai::log(myai::LogLevel::Error, "Failed to bind port " + std::to_string(myai::DEFAULT_PORT));
        return 1;
    }
    return 0;
}
