#pragma once

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#endif

#include "httplib.h"
#include "json_utils.hpp"
#include "logging.hpp"

#include <string>
#include <vector>

namespace myai {

class OllamaClient {
    std::string host;
    int port;

    std::string esc(const std::string& s) {
        std::string o;
        for (char c : s) {
            if (c == '"') o += "\\\"";
            else if (c == '\\') o += "\\\\";
            else if (c == '\n') o += "\\n";
            else if (c == '\r') o += "\\r";
            else if (c == '\t') o += "\\t";
            else o += c;
        }
        return o;
    }

    std::vector<float> parseEmbedding(const std::string& body) {
        const size_t p = body.find("\"embedding\"");
        if (p == std::string::npos) return {};
        const size_t lb = body.find('[', p);
        if (lb == std::string::npos) return {};
        size_t rb = lb + 1;
        int depth = 1;
        while (rb < body.size() && depth > 0) {
            if (body[rb] == '[') ++depth;
            else if (body[rb] == ']') --depth;
            ++rb;
        }
        if (depth != 0) return {};
        return parseVec(body.substr(lb + 1, rb - lb - 2));
    }

public:
    std::string embedModel = "nomic-embed-text";
    std::string genModel = "llama3.2";

    OllamaClient(const std::string& h = "127.0.0.1", int p = 11434)
        : host(h), port(p) {}

    bool isAvailable() {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(2, 0);
        const auto res = cli.Get("/api/tags");
        return res && res->status == 200;
    }

    std::vector<float> embed(const std::string& text) {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(3, 0);
        cli.set_read_timeout(60, 0);
        const std::string body =
            "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + esc(text) + "\"}";
        const auto res = cli.Post("/api/embeddings", body, "application/json");
        if (!res || res->status != 200) {
            log(LogLevel::Warn, "Ollama embed failed: " +
                (res ? std::to_string(res->status) : "no response"));
            return {};
        }
        return parseEmbedding(res->body);
    }

    std::string generate(const std::string& prompt) {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(3, 0);
        cli.set_read_timeout(180, 0);
        const std::string body =
            "{\"model\":\"" + genModel + "\","
            "\"prompt\":\"" + esc(prompt) + "\","
            "\"stream\":false}";
        const auto res = cli.Post("/api/generate", body, "application/json");
        if (!res || res->status != 200)
            return "ERROR: Ollama unavailable. Run: ollama serve";
        return extractStr(res->body, "response");
    }
};

}  // namespace myai
