#pragma once

#include "config.hpp"

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace myai {

inline std::string jS(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (static_cast<unsigned char>(c) < 0x20) o += ' ';
        else o += c;
    }
    return o + '"';
}

inline std::string jVec(const std::vector<float>& v) {
    std::ostringstream ss;
    ss << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) ss << ',';
        ss << std::fixed << std::setprecision(4) << v[i];
    }
    return ss.str() + ']';
}

inline std::string jsonError(const std::string& msg, int code = 400) {
    std::ostringstream ss;
    ss << "{\"error\":" << jS(msg) << ",\"code\":" << code << '}';
    return ss.str();
}

inline std::vector<float> parseVec(const std::string& s) {
    std::vector<float> v;
    std::istringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) {
        if (t.empty()) continue;
        try {
            v.push_back(std::stof(t));
        } catch (...) {}
    }
    return v;
}

inline std::string extractStr(const std::string& body, const std::string& key) {
    const size_t p = body.find('"' + key + '"');
    if (p == std::string::npos) return "";
    size_t i = body.find(':', p) + 1;
    while (i < body.size() && (body[i] == ' ' || body[i] == '\t')) ++i;
    if (i >= body.size() || body[i] != '"') return "";
    ++i;
    std::string result;
    while (i < body.size()) {
        if (body[i] == '"') break;
        if (body[i] == '\\' && i + 1 < body.size()) {
            ++i;
            switch (body[i]) {
                case '"': result += '"'; break;
                case '\\': result += '\\'; break;
                case 'n': result += '\n'; break;
                case 'r': result += '\r'; break;
                case 't': result += '\t'; break;
                default: result += body[i]; break;
            }
        } else {
            result += body[i];
        }
        ++i;
    }
    return result;
}

inline int extractInt(const std::string& body, const std::string& key, int def = 0) {
    const size_t p = body.find('"' + key + '"');
    if (p == std::string::npos) return def;
    size_t i = body.find(':', p) + 1;
    while (i < body.size() && (body[i] == ' ' || body[i] == '\t')) ++i;
    try {
        return std::stoi(body.substr(i));
    } catch (...) {
        return def;
    }
}

inline bool parseBody(const std::string& b, std::string& meta,
                      std::string& cat, std::vector<float>& emb)
{
    meta = extractStr(b, "metadata");
    cat = extractStr(b, "category");
    auto extractArr = [&](const std::string& key) -> std::vector<float> {
        const size_t p = b.find('"' + key + '"');
        if (p == std::string::npos) return {};
        const size_t lb = b.find('[', p);
        if (lb == std::string::npos) return {};
        const size_t rb = b.find(']', lb);
        if (rb == std::string::npos) return {};
        return parseVec(b.substr(lb + 1, rb - lb - 1));
    };
    emb = extractArr("embedding");
    return !meta.empty() && !emb.empty();
}

inline int clampK(int k) {
    if (k < MIN_TOP_K) return MIN_TOP_K;
    if (k > MAX_TOP_K) return MAX_TOP_K;
    return k;
}

}  // namespace myai
