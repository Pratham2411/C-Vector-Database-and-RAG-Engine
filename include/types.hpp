#pragma once

#include <functional>
#include <string>
#include <vector>

namespace myai {

struct VectorItem {
    int id;
    std::string metadata;
    std::string category;
    std::vector<float> emb;
};

struct DocItem {
    int id;
    std::string title;
    std::string text;
    std::vector<float> emb;
};

using DistFn = std::function<float(const std::vector<float>&, const std::vector<float>&)>;

}  // namespace myai
