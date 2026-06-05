#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace myai {

inline std::vector<std::string> chunkText(const std::string& text,
                                          int chunkWords = 250,
                                          int overlapWords = 30)
{
    if (chunkWords <= 0) return {};
    overlapWords = std::max(0, std::min(overlapWords, chunkWords - 1));

    std::istringstream ss(text);
    std::vector<std::string> words;
    std::string w;
    while (ss >> w) words.push_back(w);

    if (words.empty()) return {};
    if (static_cast<int>(words.size()) <= chunkWords) return {text};

    std::vector<std::string> chunks;
    const int step = std::max(1, chunkWords - overlapWords);
    for (int i = 0; i < static_cast<int>(words.size()); i += step) {
        const int end = std::min(i + chunkWords, static_cast<int>(words.size()));
        std::string chunk;
        for (int j = i; j < end; ++j) {
            if (j > i) chunk += ' ';
            chunk += words[j];
        }
        chunks.push_back(chunk);
        if (end == static_cast<int>(words.size())) break;
    }
    return chunks;
}

}  // namespace myai
