#pragma once

#include <string>

namespace myai {

constexpr int DEMO_DIMS = 16;
constexpr int DEFAULT_PORT = 8080;
constexpr int MAX_TOP_K = 50;
constexpr int MIN_TOP_K = 1;
constexpr size_t MAX_BODY_BYTES = 2 * 1024 * 1024;  // 2 MB
constexpr size_t MAX_TITLE_LEN = 512;
constexpr size_t MAX_TEXT_LEN = 500000;
constexpr size_t MAX_QUESTION_LEN = 8000;
constexpr float DOC_MAX_COSINE_DIST = 1.0f;  // cosine distance in [0, 2]

constexpr const char* PROJECT_NAME = "MY-AI";
constexpr const char* AUTHOR_NAME  = "Pratham Raj";
constexpr const char* AUTHOR_ORG   = "NIT Patna";

}  // namespace myai
