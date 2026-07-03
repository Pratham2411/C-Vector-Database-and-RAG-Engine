#pragma once

#include "types.hpp"

#include <cmath>
#include <queue>
#include <utility>
#include <vector>

namespace myai {

struct KDNode {
    VectorItem item;
    KDNode* left = nullptr;
    KDNode* right = nullptr;
    explicit KDNode(const VectorItem& v) : item(v) {}
};

class KDTree {
    KDNode* root = nullptr;
    int dims;

    void destroy(KDNode* n) {
        if (!n) return;
        destroy(n->left);
        destroy(n->right);
        delete n;
    }

    KDNode* ins(KDNode* n, const VectorItem& v, int d) {
        if (!n) return new KDNode(v);
        const int ax = d % dims;
        if (v.emb[ax] < n->item.emb[ax]) n->left = ins(n->left, v, d + 1);
        else n->right = ins(n->right, v, d + 1);
        return n;
    }

    void knn(KDNode* n, const std::vector<float>& q, int k, int d, DistFn dist,
             std::priority_queue<std::pair<float, int>>& heap) const
    {
        if (!n) return;
        const float dn = dist(q, n->item.emb);
        if (static_cast<int>(heap.size()) < k || dn < heap.top().first) {
            heap.push({dn, n->item.id});
            if (static_cast<int>(heap.size()) > k) heap.pop();
        }
        const int ax = d % dims;
        const float diff = q[ax] - n->item.emb[ax];
        KDNode* closer = diff < 0 ? n->left : n->right;
        KDNode* farther = diff < 0 ? n->right : n->left;
        knn(closer, q, k, d + 1, dist, heap);
        // Pruning is exact for Euclidean; approximate for cosine/manhattan.
        if (static_cast<int>(heap.size()) < k || std::abs(diff) < heap.top().first)
            knn(farther, q, k, d + 1, dist, heap);
    }

public:
    explicit KDTree(int d) : dims(d) {}
    ~KDTree() { destroy(root); }

    KDTree(const KDTree&) = delete;
    KDTree& operator=(const KDTree&) = delete;

    void insert(const VectorItem& v) { root = ins(root, v, 0); }

    std::vector<std::pair<float, int>> knn(
        const std::vector<float>& q, int k, DistFn dist) const
    {
        if (!root || k <= 0) return {};
        std::priority_queue<std::pair<float, int>> heap;
        knn(root, q, k, 0, dist, heap);
        std::vector<std::pair<float, int>> r;
        while (!heap.empty()) {
            r.push_back(heap.top());
            heap.pop();
        }
        std::sort(r.begin(), r.end());
        return r;
    }

    void rebuild(const std::vector<VectorItem>& items) {
        destroy(root);
        root = nullptr;
        for (const auto& v : items) insert(v);
    }
};

}  // namespace myai
