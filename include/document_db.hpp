#pragma once

#include "brute_force.hpp"
#include "config.hpp"
#include "distances.hpp"
#include "hnsw.hpp"
#include "json_utils.hpp"
#include "types.hpp"

#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace myai {

class DocumentDB {
    std::unordered_map<int, DocItem> store;
    HNSW hnsw;
    BruteForce bf;
    mutable std::shared_mutex mu;
    int nextId = 1;
    int dims = 0;

public:
    DocumentDB() : hnsw(16, 200) {}

    int insert(const std::string& title, const std::string& text,
               const std::vector<float>& emb)
    {
        std::unique_lock<std::shared_mutex> lk(mu);
        if (emb.empty()) return -1;
        if (dims == 0) dims = static_cast<int>(emb.size());
        if (static_cast<int>(emb.size()) != dims) return -1;

        DocItem item{nextId++, title, text, emb};
        store[item.id] = item;
        VectorItem vi{item.id, title, "doc", emb};
        hnsw.insert(vi, cosine);
        bf.insert(vi);
        return item.id;
    }

    std::vector<std::pair<float, DocItem>> search(
        const std::vector<float>& q, int k, float max_dist = DOC_MAX_COSINE_DIST)
    {
        std::shared_lock<std::shared_mutex> lk(mu);
        if (store.empty() || static_cast<int>(q.size()) != dims) return {};
        k = clampK(k);
        const auto raw = (store.size() < 10)
            ? bf.knn(q, k, cosine)
            : hnsw.knn(q, k, 50, cosine);
        std::vector<std::pair<float, DocItem>> out;
        for (const auto& [d, id] : raw)
            if (store.count(id) && d <= max_dist) out.push_back({d, store[id]});
        return out;
    }

    bool remove(int id) {
        std::unique_lock<std::shared_mutex> lk(mu);
        if (!store.count(id)) return false;
        store.erase(id);
        hnsw.remove(id);
        bf.remove(id);
        if (store.empty()) dims = 0;
        return true;
    }

    void rollback(const std::vector<int>& ids) {
        for (int id : ids) remove(id);
    }

    std::vector<DocItem> all() const {
        std::shared_lock<std::shared_mutex> lk(mu);
        std::vector<DocItem> r;
        r.reserve(store.size());
        for (const auto& [id, v] : store) r.push_back(v);
        return r;
    }

    size_t size() const {
        std::shared_lock<std::shared_mutex> lk(mu);
        return store.size();
    }

    int getDims() const {
        std::shared_lock<std::shared_mutex> lk(mu);
        return dims;
    }
};

}  // namespace myai
