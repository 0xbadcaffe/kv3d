#include "kv3d/api/types.hpp"
#include "kv3d/metrics/metrics.hpp"
#include "kv3d/sched/session_manager.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

// Forward-declared helpers from stream_writer.cpp
namespace kv3d::api {
std::string chunk_to_sse(const ChatCompletionChunk&);
std::string sse_done_sentinel();
std::string response_to_json(const ChatCompletionResponse&);
std::string error_to_json(const ErrorResponse&);
ChatCompletionRequest request_from_json(const std::string& body);
}  // namespace kv3d::api

namespace kv3d::api {

using json = nlohmann::json;

static int64_t unix_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

static std::string make_id() {
    static std::atomic<uint64_t> counter{0};
    return "chatcmpl-kv3d-" + std::to_string(counter.fetch_add(1, std::memory_order_relaxed));
}

// ── Route registration ────────────────────────────────────────────────────────

void register_routes(httplib::Server& srv, std::shared_ptr<SessionManager> session_mgr,
                     const std::string& api_key) {
    // Auth middleware helper
    auto check_auth = [api_key](const httplib::Request& req, httplib::Response& res) -> bool {
        if (api_key.empty()) return true;
        const auto it = req.headers.find("Authorization");
        if (it == req.headers.end() || it->second != ("Bearer " + api_key)) {
            res.status = 401;
            ErrorResponse err;
            err.error.message = "Invalid API key";
            err.error.type = "invalid_request_error";
            res.set_content(error_to_json(err), "application/json");
            return false;
        }
        return true;
    };

    // GET /health
    srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok","service":"kv3d-engine"})", "application/json");
    });

    // GET /metrics  (Prometheus text format)
    srv.Get("/metrics", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(Metrics::instance().prometheus_text(), "text/plain; version=0.0.4");
    });

    // GET /v1/models
    srv.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
        json j = {{"object", "list"},
                  {"data",
                   json::array({json{{"id", "kv3d-engine"},
                                     {"object", "model"},
                                     {"created", 0},
                                     {"owned_by", "kv3d"}}})}};
        res.set_content(j.dump(), "application/json");
    });

    // POST /v1/chat/completions
    srv.Post("/v1/chat/completions",
             [check_auth, session_mgr](const httplib::Request& req, httplib::Response& res) {
                 if (!check_auth(req, res)) return;

                 ChatCompletionRequest cr;
                 try {
                     cr = request_from_json(req.body);
                 } catch (const std::exception& e) {
                     res.status = 400;
                     ErrorResponse err;
                     err.error.message = std::string("Invalid request: ") + e.what();
                     err.error.type = "invalid_request_error";
                     res.set_content(error_to_json(err), "application/json");
                     return;
                 }

                 const std::string completion_id = make_id();
                 const int64_t created = unix_now();

                 if (cr.stream) {
                     // Server-Sent Events streaming response
                     res.set_chunked_content_provider(
                         "text/event-stream",
                         [cr, session_mgr, completion_id, created](
                             size_t /*offset*/, httplib::DataSink& sink) {
                             // First chunk: role announcement
                             ChatCompletionChunk first;
                             first.id = completion_id;
                             first.created = created;
                             first.model = cr.model;
                             first.choices = {StreamChoice{0, DeltaContent{"assistant", ""}, {}}};
                             sink.write(chunk_to_sse(first));

                             // Stream tokens
                             bool ok = true;
                             session_mgr->complete(cr, [&](std::string_view token, bool done) {
                                 ChatCompletionChunk chunk;
                                 chunk.id = completion_id;
                                 chunk.created = created;
                                 chunk.model = cr.model;
                                 StreamChoice sc;
                                 sc.delta.content = std::string(token);
                                 if (done) sc.finish_reason = "stop";
                                 chunk.choices = {sc};
                                 ok = sink.write(chunk_to_sse(chunk));
                             });

                             sink.write(sse_done_sentinel());
                             sink.done();
                             return true;
                         });
                 } else {
                     // Full response
                     std::string content;
                     try {
                         content = session_mgr->complete(cr);
                     } catch (const std::exception& e) {
                         res.status = 500;
                         ErrorResponse err;
                         err.error.message = e.what();
                         err.error.type = "server_error";
                         res.set_content(error_to_json(err), "application/json");
                         return;
                     }

                     ChatCompletionResponse resp;
                     resp.id = completion_id;
                     resp.created = created;
                     resp.model = cr.model;
                     resp.choices = {Choice{0, {"assistant", content}, "stop"}};
                     resp.usage.completion_tokens = static_cast<int>(content.size() / 4);
                     resp.usage.prompt_tokens = static_cast<int>(cr.messages.size() * 10);
                     resp.usage.total_tokens =
                         resp.usage.prompt_tokens + resp.usage.completion_tokens;

                     res.set_content(response_to_json(resp), "application/json");
                 }
             });
}

}  // namespace kv3d::api
