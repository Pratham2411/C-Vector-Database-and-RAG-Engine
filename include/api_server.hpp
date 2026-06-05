#pragma once

#include "document_db.hpp"
#include "httplib.h"
#include "ollama_client.hpp"
#include "vector_db.hpp"

namespace myai {

void registerRoutes(httplib::Server& svr, VectorDB& db, DocumentDB& docDB, OllamaClient& ollama);

}  // namespace myai
