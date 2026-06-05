#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#endif

#include "chunker.hpp"
#include "config.hpp"
#include "demo_data.hpp"
#include "distances.hpp"
#include "document_db.hpp"
#include "http_common.hpp"
#include "json_utils.hpp"
#include "logging.hpp"
#include "ollama_client.hpp"
#include "vector_db.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace myai {

static bool parseId(const std::string& s, int& out) {
    try {
        out = std::stoi(s);
        return out > 0;
    } catch (...) {
        return false;
    }
}

static std::string readIndexHtml() {
    std::ifstream f("index.html");
    if (!f.is_open()) return "";
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

void registerRoutes(httplib::Server& svr, VectorDB& db, DocumentDB& docDB, OllamaClient& ollama) {
    svr.set_payload_max_length(MAX_BODY_BYTES);

    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        cors(res);
        res.status = 204;
    });

    svr.Get("/search", [&](const httplib::Request& req, httplib::Response& res) {
        const auto q = parseVec(req.get_param_value("v"));
        if (static_cast<int>(q.size()) != DEMO_DIMS) {
            jsonResponse(res, jsonError("Expected " + std::to_string(DEMO_DIMS) + "D query vector"), 400);
            return;
        }
        int k = MIN_TOP_K;
        try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        k = clampK(k);

        auto metric = req.get_param_value("metric");
        if (metric.empty()) metric = "cosine";
        if (!isValidMetric(metric)) {
            jsonResponse(res, jsonError("Invalid metric"), 400);
            return;
        }

        auto algo = req.get_param_value("algo");
        if (algo.empty()) algo = "hnsw";
        if (!isValidAlgo(algo)) {
            jsonResponse(res, jsonError("Invalid algorithm"), 400);
            return;
        }

        const auto out = db.search(q, k, metric, algo);
        std::ostringstream ss;
        ss << "{\"results\":[";
        for (size_t i = 0; i < out.hits.size(); ++i) {
            if (i) ss << ',';
            const auto& h = out.hits[i];
            ss << "{\"id\":" << h.id
               << ",\"metadata\":" << jS(h.meta)
               << ",\"category\":" << jS(h.cat)
               << ",\"distance\":" << std::fixed << std::setprecision(6) << h.dist
               << ",\"embedding\":" << jVec(h.emb) << '}';
        }
        ss << "],\"latencyUs\":" << out.us
           << ",\"algo\":" << jS(out.algo)
           << ",\"metric\":" << jS(out.metric);
        if (!out.warning.empty()) ss << ",\"warning\":" << jS(out.warning);
        ss << '}';
        jsonResponse(res, ss.str());
    });

    svr.Post("/insert", [&](const httplib::Request& req, httplib::Response& res) {
        std::string meta, cat;
        std::vector<float> emb;
        if (!parseBody(req.body, meta, cat, emb) || static_cast<int>(emb.size()) != DEMO_DIMS) {
            jsonResponse(res, jsonError("Invalid body: metadata, category, and " +
                std::to_string(DEMO_DIMS) + "D embedding required"), 400);
            return;
        }
        if (meta.size() > MAX_TITLE_LEN) {
            jsonResponse(res, jsonError("Metadata too long"), 400);
            return;
        }
        const int id = db.insert(meta, cat.empty() ? "custom" : cat, emb, getDistFn("cosine"));
        jsonResponse(res, "{\"id\":" + std::to_string(id) + "}", 201);
    });

    svr.Delete(R"(/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        int id = 0;
        if (!parseId(req.matches[1], id)) {
            jsonResponse(res, jsonError("Invalid id"), 400);
            return;
        }
        const bool ok = db.remove(id);
        jsonResponse(res, "{\"ok\":" + std::string(ok ? "true" : "false") + "}", ok ? 200 : 404);
    });

    svr.Get("/items", [&](const httplib::Request&, httplib::Response& res) {
        const auto items = db.all();
        std::ostringstream ss;
        ss << '[';
        for (size_t i = 0; i < items.size(); ++i) {
            if (i) ss << ',';
            const auto& v = items[i];
            ss << "{\"id\":" << v.id
               << ",\"metadata\":" << jS(v.metadata)
               << ",\"category\":" << jS(v.category)
               << ",\"embedding\":" << jVec(v.emb) << '}';
        }
        ss << ']';
        jsonResponse(res, ss.str());
    });

    svr.Get("/benchmark", [&](const httplib::Request& req, httplib::Response& res) {
        const auto q = parseVec(req.get_param_value("v"));
        if (static_cast<int>(q.size()) != DEMO_DIMS) {
            jsonResponse(res, jsonError("Expected " + std::to_string(DEMO_DIMS) + "D query vector"), 400);
            return;
        }
        int k = 5;
        try { k = std::stoi(req.get_param_value("k")); } catch (...) {}
        auto metric = req.get_param_value("metric");
        if (metric.empty()) metric = "cosine";
        if (!isValidMetric(metric)) {
            jsonResponse(res, jsonError("Invalid metric"), 400);
            return;
        }
        const auto b = db.benchmark(q, k, metric);
        std::ostringstream ss;
        ss << "{\"bruteforceUs\":" << b.bfUs
           << ",\"kdtreeUs\":" << b.kdUs
           << ",\"hnswUs\":" << b.hnswUs
           << ",\"itemCount\":" << b.n << '}';
        jsonResponse(res, ss.str());
    });

    svr.Get("/hnsw-info", [&](const httplib::Request&, httplib::Response& res) {
        const auto gi = db.hnswInfo();
        std::ostringstream ss;
        ss << "{\"topLayer\":" << gi.topLayer << ",\"nodeCount\":" << gi.nodeCount
           << ",\"nodesPerLayer\":[";
        for (size_t i = 0; i < gi.nodesPerLayer.size(); ++i) {
            if (i) ss << ',';
            ss << gi.nodesPerLayer[i];
        }
        ss << "],\"edgesPerLayer\":[";
        for (size_t i = 0; i < gi.edgesPerLayer.size(); ++i) {
            if (i) ss << ',';
            ss << gi.edgesPerLayer[i];
        }
        ss << "],\"nodes\":[";
        for (size_t i = 0; i < gi.nodes.size(); ++i) {
            if (i) ss << ',';
            const auto& n = gi.nodes[i];
            ss << "{\"id\":" << n.id << ",\"metadata\":" << jS(n.metadata)
               << ",\"category\":" << jS(n.category) << ",\"maxLyr\":" << n.maxLyr << '}';
        }
        ss << "],\"edges\":[";
        for (size_t i = 0; i < gi.edges.size(); ++i) {
            if (i) ss << ',';
            const auto& e = gi.edges[i];
            ss << "{\"src\":" << e.src << ",\"dst\":" << e.dst << ",\"lyr\":" << e.lyr << '}';
        }
        ss << "]}";
        jsonResponse(res, ss.str());
    });

    svr.Post("/doc/insert", [&](const httplib::Request& req, httplib::Response& res) {
        auto title = extractStr(req.body, "title");
        auto text = extractStr(req.body, "text");
        if (title.empty() || text.empty()) {
            jsonResponse(res, jsonError("Need title and text"), 400);
            return;
        }
        if (title.size() > MAX_TITLE_LEN || text.size() > MAX_TEXT_LEN) {
            jsonResponse(res, jsonError("Title or text exceeds size limit"), 400);
            return;
        }

        const auto chunks = chunkText(text, 250, 30);
        std::vector<int> ids;

        for (int i = 0; i < static_cast<int>(chunks.size()); ++i) {
            const auto emb = ollama.embed(chunks[i]);
            if (emb.empty()) {
                docDB.rollback(ids);
                jsonResponse(res, jsonError(
                    "Ollama unavailable. Install from https://ollama.com and run: "
                    "ollama pull nomic-embed-text && ollama pull llama3.2"), 503);
                return;
            }
            const std::string chunkTitle = (chunks.size() > 1)
                ? title + " [" + std::to_string(i + 1) + "/" + std::to_string(chunks.size()) + "]"
                : title;
            const int id = docDB.insert(chunkTitle, chunks[i], emb);
            if (id < 0) {
                docDB.rollback(ids);
                jsonResponse(res, jsonError("Embedding dimension mismatch"), 500);
                return;
            }
            ids.push_back(id);
        }

        std::ostringstream ss;
        ss << "{\"ids\":[";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) ss << ',';
            ss << ids[i];
        }
        ss << "],\"chunks\":" << chunks.size()
           << ",\"dims\":" << docDB.getDims() << '}';
        jsonResponse(res, ss.str(), 201);
    });

    svr.Delete(R"(/doc/delete/(\d+))", [&](const httplib::Request& req, httplib::Response& res) {
        int id = 0;
        if (!parseId(req.matches[1], id)) {
            jsonResponse(res, jsonError("Invalid id"), 400);
            return;
        }
        const bool ok = docDB.remove(id);
        jsonResponse(res, "{\"ok\":" + std::string(ok ? "true" : "false") + "}", ok ? 200 : 404);
    });

    svr.Get("/doc/list", [&](const httplib::Request&, httplib::Response& res) {
        const auto docs = docDB.all();
        std::ostringstream ss;
        ss << '[';
        for (size_t i = 0; i < docs.size(); ++i) {
            if (i) ss << ',';
            std::string preview = docs[i].text.substr(0, 120);
            if (docs[i].text.size() > 120) preview += "...";
            ss << "{\"id\":" << docs[i].id
               << ",\"title\":" << jS(docs[i].title)
               << ",\"preview\":" << jS(preview)
               << ",\"words\":" << static_cast<int>(
                      std::count(docs[i].text.begin(), docs[i].text.end(), ' ') + 1)
               << '}';
        }
        ss << ']';
        jsonResponse(res, ss.str());
    });

    svr.Post("/doc/search", [&](const httplib::Request& req, httplib::Response& res) {
        const auto question = extractStr(req.body, "question");
        int k = clampK(extractInt(req.body, "k", 3));
        if (question.empty()) {
            jsonResponse(res, jsonError("Need question"), 400);
            return;
        }
        if (question.size() > MAX_QUESTION_LEN) {
            jsonResponse(res, jsonError("Question too long"), 400);
            return;
        }

        const auto qEmb = ollama.embed(question);
        if (qEmb.empty()) {
            jsonResponse(res, jsonError("Ollama unavailable"), 503);
            return;
        }

        const auto hits = docDB.search(qEmb, k);
        std::ostringstream ss;
        ss << "{\"contexts\":[";
        for (size_t i = 0; i < hits.size(); ++i) {
            if (i) ss << ',';
            ss << "{\"id\":" << hits[i].second.id
               << ",\"title\":" << jS(hits[i].second.title)
               << ",\"distance\":" << std::fixed << std::setprecision(4) << hits[i].first << '}';
        }
        ss << "]}";
        jsonResponse(res, ss.str());
    });

    svr.Post("/doc/ask", [&](const httplib::Request& req, httplib::Response& res) {
        const auto question = extractStr(req.body, "question");
        int k = clampK(extractInt(req.body, "k", 3));
        if (question.empty()) {
            jsonResponse(res, jsonError("Need question"), 400);
            return;
        }
        if (question.size() > MAX_QUESTION_LEN) {
            jsonResponse(res, jsonError("Question too long"), 400);
            return;
        }

        const auto qEmb = ollama.embed(question);
        if (qEmb.empty()) {
            jsonResponse(res, jsonError("Ollama unavailable"), 503);
            return;
        }

        const auto hits = docDB.search(qEmb, k);

        std::ostringstream ctx;
        for (int i = 0; i < static_cast<int>(hits.size()); ++i) {
            ctx << "[" << (i + 1) << "] " << hits[i].second.title << ":\n"
                << hits[i].second.text << "\n\n";
        }

        const std::string prompt =
            "You are a helpful assistant built for MY-AI by Pratham Raj (NIT Patna). "
            "Answer the user's question directly using the provided context when relevant. "
            "If context is insufficient, answer from general knowledge. "
            "Do not mention 'context' or 'provided text'.\n\n"
            "Context:\n" + ctx.str() +
            "Question: " + question + "\n\n"
            "Answer:";

        const auto answer = ollama.generate(prompt);

        std::ostringstream ss;
        ss << "{\"answer\":" << jS(answer)
           << ",\"model\":" << jS(ollama.genModel)
           << ",\"contexts\":[";
        for (size_t i = 0; i < hits.size(); ++i) {
            if (i) ss << ',';
            ss << "{\"id\":" << hits[i].second.id
               << ",\"title\":" << jS(hits[i].second.title)
               << ",\"text\":" << jS(hits[i].second.text)
               << ",\"distance\":" << std::fixed << std::setprecision(4) << hits[i].first << '}';
        }
        ss << "],\"docCount\":" << docDB.size() << '}';
        jsonResponse(res, ss.str());
    });

    svr.Get("/status", [&](const httplib::Request&, httplib::Response& res) {
        const bool up = ollama.isAvailable();
        std::ostringstream ss;
        ss << "{\"project\":" << jS(PROJECT_NAME)
           << ",\"author\":" << jS(AUTHOR_NAME)
           << ",\"organization\":" << jS(AUTHOR_ORG)
           << ",\"ollamaAvailable\":" << (up ? "true" : "false")
           << ",\"embedModel\":" << jS(ollama.embedModel)
           << ",\"genModel\":" << jS(ollama.genModel)
           << ",\"docCount\":" << docDB.size()
           << ",\"docDims\":" << docDB.getDims()
           << ",\"demoDims\":" << DEMO_DIMS
           << ",\"demoCount\":" << db.size() << '}';
        jsonResponse(res, ss.str());
    });

    svr.Get("/stats", [&](const httplib::Request&, httplib::Response& res) {
        std::ostringstream ss;
        ss << "{\"count\":" << db.size()
           << ",\"dims\":" << DEMO_DIMS
           << ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
           << ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";
        jsonResponse(res, ss.str());
    });

    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        static const std::string html = readIndexHtml();
        if (html.empty()) {
            res.status = 404;
            res.set_content("index.html not found", "text/plain");
            return;
        }
        cors(res);
        res.set_content(html, "text/html");
    });
}

}  // namespace myai
