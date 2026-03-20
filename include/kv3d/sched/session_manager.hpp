#pragma once

#include "kv3d/api/types.hpp"
#include "kv3d/core/canonical_prompt.hpp"
#include "kv3d/core/guardrails.hpp"
#include "kv3d/kv/delta_codec.hpp"
#include "kv3d/kv/kv_block.hpp"
#include "kv3d/kv/prefix_store.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace kv3d {

enum class SessionState { Active, Paused, Expired };

struct Session {
    uint64_t session_id;
    uint64_t family_id;
    std::string model_id;
    SessionState state{SessionState::Active};
    std::chrono::steady_clock::time_point last_active;
    std::optional<SessionDelta> delta;  // nullopt when fully on shared prefix
    std::vector<ChatMessage> history;
    uint32_t total_tokens{0};
};

struct SessionManagerConfig {
    size_t max_active_sessions{256};
    size_t max_prefix_store_bytes{static_cast<size_t>(8) * 1024 * 1024 * 1024};  // 8 GiB RAM
    std::chrono::seconds session_ttl{3600};
    PrefixExtractionOptions prefix_opts{};
    FallbackPolicy fallback_policy{};
};

using TokenCallback = std::function<void(std::string_view token, bool done)>;

/// Central coordinator: receives requests, manages prefix sharing,
/// invokes the backend, streams tokens back, and tracks sessions.
class SessionManager {
public:
    explicit SessionManager(SessionManagerConfig config = {});
    ~SessionManager();

    SessionManager(const SessionManager&) = delete;
    SessionManager& operator=(const SessionManager&) = delete;

    /// Process a chat completion request.
    /// Calls `on_token` for each generated token (streaming mode).
    /// Returns the final response text.
    std::string complete(const api::ChatCompletionRequest& req,
                         TokenCallback on_token = nullptr);

    /// Pause a session: compress its delta and free GPU memory.
    bool pause(uint64_t session_id);

    /// Resume a paused session: restore KV from warm cache.
    bool resume(uint64_t session_id);

    /// Statistics for observability.
    struct Stats {
        uint64_t prefix_hits{0};
        uint64_t prefix_misses{0};
        uint64_t fallback_events{0};
        size_t active_sessions{0};
        size_t prefix_store_bytes{0};
    };
    [[nodiscard]] Stats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace kv3d
