#pragma once

#include "types.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace myai {

class BruteForce {
public:
    std::vector<VectorItem> items;

    void insert(const VectorItem& v) { items.push_back(v); }

    std::vector<std::pair<float, int>> knn(
        const std::vector<float>& q, int k, DistFn dist) const
    {
        if (k <= 0 || items.empty()) return {};
        std::vector<std::pair<float, int>> r;
        r.reserve(items.size());
        for (const auto& v : items) r.push_back({dist(q, v.emb), v.id});
        std::sort(r.begin(), r.end());
        if (static_cast<int>(r.size()) > k) r.resize(static_cast<size_t>(k));
        return r;
    }

    void remove(int id) {
        items.erase(
            std::remove_if(items.begin(), items.end(),
                [id](const VectorItem& v) { return v.id == id; }),
            items.end());
    }
};

}  // namespace myai
