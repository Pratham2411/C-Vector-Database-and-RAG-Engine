#pragma once

#include "types.hpp"

#include <cmath>
#include <string>

namespace myai {

inline float euclidean(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0;
    const int n = static_cast<int>(std::min(a.size(), b.size()));
    for (int i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

inline float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0, na = 0, nb = 0;
    const int n = static_cast<int>(std::min(a.size(), b.size()));
    for (int i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na < 1e-9f || nb < 1e-9f) return 1.0f;
    return 1.0f - dot / (std::sqrt(na) * std::sqrt(nb));
}

inline float manhattan(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0;
    const int n = static_cast<int>(std::min(a.size(), b.size()));
    for (int i = 0; i < n; ++i) s += std::abs(a[i] - b[i]);
    return s;
}

inline DistFn getDistFn(const std::string& metric) {
    if (metric == "cosine") return cosine;
    if (metric == "manhattan") return manhattan;
    return euclidean;
}

inline bool isValidMetric(const std::string& metric) {
    return metric == "cosine" || metric == "euclidean" || metric == "manhattan";
}

inline bool isValidAlgo(const std::string& algo) {
    return algo == "hnsw" || algo == "kdtree" || algo == "bruteforce";
}

}  // namespace myai
