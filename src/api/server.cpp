#include "kv3d/api/server.hpp"
#include "kv3d/sched/session_manager.hpp"

#include <httplib.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>

// Forward declaration from routes.cpp
namespace kv3d::api {
void register_routes(httplib::Server& srv, std::shared_ptr<SessionManager> session_mgr,
                     const std::string& api_key);
}  // namespace kv3d::api

namespace kv3d::api {

struct Server::Impl {
    ServerConfig config;
    std::shared_ptr<SessionManager> session_mgr;
    httplib::Server http;
    std::atomic<bool> running{false};

    explicit Impl(ServerConfig cfg, std::shared_ptr<SessionManager> mgr)
        : config(std::move(cfg)), session_mgr(std::move(mgr)) {}
};

Server::Server(ServerConfig config, std::shared_ptr<SessionManager> session_mgr)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(session_mgr))) {
    if (impl_->config.enable_cors) {
        impl_->http.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Allow-Headers", "Authorization, Content-Type"},
             {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"}});
        impl_->http.Options(".*", [](const httplib::Request&, httplib::Response& res) {
            res.status = 204;
        });
    }

    impl_->http.set_read_timeout(120);   // 2 min for long completions
    impl_->http.set_write_timeout(120);
    impl_->http.new_task_queue = [this] {
        return new httplib::ThreadPool(impl_->config.thread_count);
    };

    register_routes(impl_->http, impl_->session_mgr, impl_->config.api_key);
}

Server::~Server() { stop(); }

void Server::run() {
    impl_->running.store(true);
    impl_->http.listen(impl_->config.host, impl_->config.port);
    impl_->running.store(false);
}

void Server::stop() {
    if (impl_->running.load()) {
        impl_->http.stop();
    }
}

bool Server::is_running() const noexcept { return impl_->running.load(); }

}  // namespace kv3d::api
