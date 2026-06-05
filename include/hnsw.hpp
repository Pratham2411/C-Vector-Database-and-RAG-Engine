#pragma once

#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <queue>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace myai {

class HNSW {
    struct Node {
        VectorItem item;
        int maxLyr;
        std::vector<std::vector<int>> nbrs;
    };

    std::unordered_map<int, Node> G;
    int M, M0, ef_build;
    float mL;
    int topLayer = -1;
    int entryPt = -1;
    std::mt19937 rng;

    int randLevel() {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        return static_cast<int>(std::floor(-std::log(u(rng)) * mL));
    }

    void recomputeEntryPoint(int removedId) {
        entryPt = -1;
        topLayer = -1;
        for (const auto& [nid, nd] : G) {
            if (nid == removedId) continue;
            if (entryPt == -1) entryPt = nid;
            topLayer = std::max(topLayer, nd.maxLyr);
        }
    }

    std::vector<std::pair<float, int>> searchLayer(
        const std::vector<float>& q, int ep, int ef, int lyr, DistFn dist)
    {
        std::unordered_map<int, bool> vis;
        std::priority_queue<std::pair<float, int>,
            std::vector<std::pair<float, int>>, std::greater<>> cands;
        std::priority_queue<std::pair<float, int>> found;

        const float d0 = dist(q, G.at(ep).item.emb);
        vis[ep] = true;
        cands.push({d0, ep});
        found.push({d0, ep});

        while (!cands.empty()) {
            auto [cd, cid] = cands.top();
            cands.pop();
            if (static_cast<int>(found.size()) >= ef && cd > found.top().first) break;
            if (lyr >= static_cast<int>(G.at(cid).nbrs.size())) continue;
            for (int nid : G.at(cid).nbrs[lyr]) {
                if (vis[nid] || !G.count(nid)) continue;
                vis[nid] = true;
                const float nd = dist(q, G.at(nid).item.emb);
                if (static_cast<int>(found.size()) < ef || nd < found.top().first) {
                    cands.push({nd, nid});
                    found.push({nd, nid});
                    if (static_cast<int>(found.size()) > ef) found.pop();
                }
            }
        }

        std::vector<std::pair<float, int>> res;
        while (!found.empty()) {
            res.push_back(found.top());
            found.pop();
        }
        std::sort(res.begin(), res.end());
        return res;
    }

    std::vector<int> selectNbrs(std::vector<std::pair<float, int>>& cands, int maxM) {
        std::vector<int> r;
        for (int i = 0; i < std::min(static_cast<int>(cands.size()), maxM); ++i)
            r.push_back(cands[i].second);
        return r;
    }

public:
    HNSW(int m = 16, int efBuild = 200)
        : M(m), M0(2 * m), ef_build(efBuild),
          mL(1.0f / std::log(static_cast<float>(m))), rng(42) {}

    void insert(const VectorItem& item, DistFn dist) {
        const int id = item.id;
        const int lvl = randLevel();
        G[id] = {item, lvl, std::vector<std::vector<int>>(lvl + 1)};

        if (entryPt == -1) {
            entryPt = id;
            topLayer = lvl;
            return;
        }

        int ep = entryPt;
        for (int lc = topLayer; lc > lvl; --lc) {
            if (lc < static_cast<int>(G.at(ep).nbrs.size())) {
                auto W = searchLayer(item.emb, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        for (int lc = std::min(topLayer, lvl); lc >= 0; --lc) {
            auto W = searchLayer(item.emb, ep, ef_build, lc, dist);
            const int maxM = (lc == 0) ? M0 : M;
            auto sel = selectNbrs(W, maxM);
            G[id].nbrs[lc] = sel;

            for (int nid : sel) {
                if (!G.count(nid)) continue;
                if (static_cast<int>(G[nid].nbrs.size()) <= lc)
                    G[nid].nbrs.resize(lc + 1);
                auto& conn = G[nid].nbrs[lc];
                conn.push_back(id);
                if (static_cast<int>(conn.size()) > maxM) {
                    std::vector<std::pair<float, int>> ds;
                    for (int c : conn)
                        if (G.count(c))
                            ds.push_back({dist(G[nid].item.emb, G[c].item.emb), c});
                    std::sort(ds.begin(), ds.end());
                    conn.clear();
                    for (int i = 0; i < maxM && i < static_cast<int>(ds.size()); ++i)
                        conn.push_back(ds[i].second);
                }
            }
            if (!W.empty()) ep = W[0].second;
        }
        if (lvl > topLayer) {
            topLayer = lvl;
            entryPt = id;
        }
    }

    std::vector<std::pair<float, int>> knn(
        const std::vector<float>& q, int k, int ef, DistFn dist)
    {
        if (entryPt == -1 || k <= 0) return {};
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; --lc) {
            if (lc < static_cast<int>(G.at(ep).nbrs.size())) {
                auto W = searchLayer(q, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        auto W = searchLayer(q, ep, std::max(ef, k), 0, dist);
        if (static_cast<int>(W.size()) > k) W.resize(static_cast<size_t>(k));
        return W;
    }

    void remove(int id) {
        if (!G.count(id)) return;
        for (auto& [nid, nd] : G)
            for (auto& layer : nd.nbrs)
                layer.erase(std::remove(layer.begin(), layer.end(), id), layer.end());
        if (entryPt == id) recomputeEntryPoint(id);
        G.erase(id);
    }

    struct GraphInfo {
        int topLayer, nodeCount;
        std::vector<int> nodesPerLayer, edgesPerLayer;
        struct NV { int id; std::string metadata, category; int maxLyr; };
        struct EV { int src, dst, lyr; };
        std::vector<NV> nodes;
        std::vector<EV> edges;
    };

    GraphInfo getInfo() {
        GraphInfo gi;
        gi.topLayer = topLayer;
        gi.nodeCount = static_cast<int>(G.size());
        const int maxL = std::max(topLayer + 1, 1);
        gi.nodesPerLayer.assign(maxL, 0);
        gi.edgesPerLayer.assign(maxL, 0);
        for (const auto& [id, nd] : G) {
            gi.nodes.push_back({id, nd.item.metadata, nd.item.category, nd.maxLyr});
            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; ++lc) {
                gi.nodesPerLayer[lc]++;
                if (lc < static_cast<int>(nd.nbrs.size()))
                    for (int nid : nd.nbrs[lc])
                        if (id < nid) {
                            gi.edgesPerLayer[lc]++;
                            gi.edges.push_back({id, nid, lc});
                        }
            }
        }
        return gi;
    }

    size_t size() const { return G.size(); }
};

}  // namespace myai
