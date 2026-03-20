#pragma once

#include <memory>
#include <string>

namespace kv3d {

class SessionManager;

namespace api {

struct ServerConfig {
    std::string host{"127.0.0.1"};
    int port{8080};
    int thread_count{4};
    bool enable_cors{true};
    std::string api_key;           // empty = no auth
    std::string log_level{"info"}; // debug | info | warn | error
};

/// HTTP server wrapping cpp-httplib.
/// Exposes OpenAI-compatible endpoints plus kv3d extensions.
class Server {
public:
    explicit Server(ServerConfig config, std::shared_ptr<SessionManager> session_mgr);
    ~Server();

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    /// Start listening (blocks until stop() is called).
    void run();

    /// Signal the server to shut down.
    void stop();

    [[nodiscard]] bool is_running() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace api
}  // namespace kv3d
