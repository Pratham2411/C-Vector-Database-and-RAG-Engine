#pragma once

#include "brute_force.hpp"
#include "config.hpp"
#include "distances.hpp"
#include "hnsw.hpp"
#include "json_utils.hpp"
#include "kd_tree.hpp"
#include "types.hpp"

#include <chrono>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace myai {

class VectorDB {
    std::unordered_map<int, VectorItem> store;
    BruteForce bf;
    KDTree kdt;
    HNSW hnsw;
    std::mutex mu;
    int nextId = 1;

public:
    const int dims;

    explicit VectorDB(int d) : kdt(d), hnsw(16, 200), dims(d) {}

    int insert(const std::string& meta, const std::string& cat,
               const std::vector<float>& emb, DistFn dist)
    {
        std::lock_guard<std::mutex> lk(mu);
        VectorItem v{nextId++, meta, cat, emb};
        store[v.id] = v;
        bf.insert(v);
        kdt.insert(v);
        hnsw.insert(v, dist);
        return v.id;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu);
        if (!store.count(id)) return false;
        store.erase(id);
        bf.remove(id);
        hnsw.remove(id);
        std::vector<VectorItem> rem;
        rem.reserve(store.size());
        for (const auto& [i, v] : store) rem.push_back(v);
        kdt.rebuild(rem);
        return true;
    }

    struct Hit {
        int id;
        std::string meta, cat;
        std::vector<float> emb;
        float dist;
    };

    struct SearchOut {
        std::vector<Hit> hits;
        long long us;
        std::string algo, metric;
        std::string warning;
    };

    SearchOut search(const std::vector<float>& q, int k,
                     const std::string& metric, const std::string& algo)
    {
        std::lock_guard<std::mutex> lk(mu);
        k = clampK(k);
        const auto dfn = getDistFn(metric);
        const auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<std::pair<float, int>> raw;
        if (algo == "bruteforce") raw = bf.knn(q, k, dfn);
        else if (algo == "kdtree") raw = kdt.knn(q, k, dfn);
        else raw = hnsw.knn(q, k, 50, dfn);

        const long long us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();

        SearchOut out;
        out.us = us;
        out.algo = algo;
        out.metric = metric;
        if (algo == "kdtree" && metric != "euclidean")
            out.warning = "KD-Tree pruning is exact for Euclidean only; results may be approximate.";
        for (const auto& [d, id] : raw)
            if (store.count(id))
                out.hits.push_back({id, store[id].metadata, store[id].category, store[id].emb, d});
        return out;
    }

    struct BenchOut { long long bfUs, kdUs, hnswUs; int n; };

    BenchOut benchmark(const std::vector<float>& q, int k, const std::string& metric) {
        std::lock_guard<std::mutex> lk(mu);
        k = clampK(k);
        const auto dfn = getDistFn(metric);
        const auto time = [&](auto fn) -> long long {
            const auto t = std::chrono::high_resolution_clock::now();
            fn();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - t).count();
        };
        return {
            time([&] { bf.knn(q, k, dfn); }),
            time([&] { kdt.knn(q, k, dfn); }),
            time([&] { hnsw.knn(q, k, 50, dfn); }),
            static_cast<int>(store.size())
        };
    }

    std::vector<VectorItem> all() {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<VectorItem> r;
        r.reserve(store.size());
        for (const auto& [id, v] : store) r.push_back(v);
        return r;
    }

    HNSW::GraphInfo hnswInfo() {
        std::lock_guard<std::mutex> lk(mu);
        return hnsw.getInfo();
    }

    size_t size() {
        std::lock_guard<std::mutex> lk(mu);
        return store.size();
    }
};

}  // namespace myai
