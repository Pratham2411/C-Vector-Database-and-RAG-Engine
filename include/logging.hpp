#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace myai {

enum class LogLevel { Info, Warn, Error };

inline std::string timestamp() {
    using clock = std::chrono::system_clock;
    auto t = clock::to_time_t(clock::now());
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

inline void log(LogLevel level, const std::string& msg) {
    const char* tag = level == LogLevel::Info ? "INFO"
                    : level == LogLevel::Warn ? "WARN" : "ERROR";
    std::cerr << "[" << timestamp() << "] [" << tag << "] " << msg << std::endl;
}

}  // namespace myai
