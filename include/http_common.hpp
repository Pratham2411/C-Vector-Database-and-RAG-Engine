#pragma once

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#endif

#include "httplib.h"

namespace myai {

inline void cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

inline void jsonResponse(httplib::Response& res, const std::string& body, int status = 200) {
    cors(res);
    res.status = status;
    res.set_content(body, "application/json");
}

}  // namespace myai
