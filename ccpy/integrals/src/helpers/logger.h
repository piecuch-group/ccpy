// Logger.h
#pragma once
#include <iostream>
#include <sstream>
#include <string>
#include <mutex>

namespace ccpy {
class Logger {
  public:
    enum Level { NONE = 0, WARNING = 1, ESSENTIAL = 2, INFO1 = 3, INFO2 = 4, DEBUG = 5 };

    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void setLevel(Level level) {
        std::lock_guard<std::mutex> lock(mutex_);
        current_level_ = level;
    }

    Level getLevel() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_level_;
    }

    void log(Level level, const std::string& message) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (level <= current_level_) {
            auto& stream = std::cout;
            stream << message << std::endl;
        }
    }

  private:
    Logger() = default;
    Level current_level_ = INFO1; // Default to INFO level
    mutable std::mutex mutex_;

    std::string levelToString(Level level) const {
        switch (level) {
        case NONE:
            return "NONE";
        case WARNING:
            return "WARNING";
        case ESSENTIAL:
            return "ESSENTIAL";
        case INFO1:
            return "INFO1";
        case INFO2:
            return "INFO2";
        case DEBUG:
            return "DEBUG";
        default:
            return "UNKNOWN";
        }
    }
};

class LogStream {
  private:
    std::ostringstream oss_;
    Logger::Level level_;
    bool should_log_;

  public:
    LogStream(Logger::Level level)
        : level_(level), should_log_(level <= Logger::getInstance().getLevel()) {}

    template <typename T> LogStream& operator<<(const T& value) {
        if (should_log_) {
            oss_ << value;
        }
        return *this;
    }

    ~LogStream() {
        if (should_log_) {
            Logger::getInstance().log(level_, oss_.str());
        }
    }

    LogStream(const LogStream&) = delete;
    LogStream& operator=(const LogStream&) = delete;

    LogStream(LogStream&& other) noexcept
        : oss_(std::move(other.oss_)), level_(other.level_), should_log_(other.should_log_) {
        other.should_log_ = false;
    }
};

#define LOG(level) LogStream(static_cast<Logger::Level>(level))
#define LOG_WARNING LogStream(Logger::WARNING)
#define LOG_ESSENTIAL LogStream(Logger::ESSENTIAL)
#define LOG_INFO1 LogStream(Logger::INFO1)
#define LOG_INFO2 LogStream(Logger::INFO2)
#define LOG_DEBUG LogStream(Logger::DEBUG)
} // namespace ccpy