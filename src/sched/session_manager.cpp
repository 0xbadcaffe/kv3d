#include "kv3d/sched/session_manager.hpp"

#include "kv3d/core/canonical_prompt.hpp"
#include "kv3d/core/prefix_hash.hpp"
#include "kv3d/metrics/metrics.hpp"
#include "kv3d/storage/ram_cache.hpp"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace kv3d {

// ── Stub inference backend ────────────────────────────────────────────────────
// This is replaced by a real llama.cpp backend once the submodule is present.
// The stub generates plausible-looking token sequences for integration testing.

namespace backend {

static std::string generate_stub(const std::string& prompt_summary, int max_tokens,
                                 TokenCallback on_token) {
    const std::vector<std::string> vocab = {
        "The ", "answer ", "is ", "based ", "on ", "the ", "context ", "provided. ",
        "I ", "can ", "help ", "you ", "with ", "that. ", "Here ", "are ", "some ",
        "key ", "points: ", "First, ", "second, ", "finally, ",
    };

    std::mt19937 rng(std::hash<std::string>{}(prompt_summary));
    std::uniform_int_distribution<size_t> dist(0, vocab.size() - 1);

    std::string full;
    const int n = std::min(max_tokens / 4, 20);
    for (int i = 0; i < n; ++i) {
        const std::string& tok = vocab[dist(rng)];
        full += tok;
        if (on_token) on_token(tok, i == n - 1);
    }
    return full;
}

}  // namespace backend

// ── SessionManager::Impl ──────────────────────────────────────────────────────

struct SessionManager::Impl {
    SessionManagerConfig config;
    PrefixStore prefix_store;
    RAMCache ram_cache;
    Guardrails guardrails;
    DeltaCodec codec;

    mutable std::mutex sessions_mutex;
    std::unordered_map<uint64_t, Session> sessions;
    std::atomic<uint64_t> next_session_id{1};

    explicit Impl(SessionManagerConfig cfg)
        : config(std::move(cfg)),
          ram_cache(config.max_prefix_store_bytes),
          guardrails(config.fallback_policy) {}

    uint64_t allocate_session_id() noexcept {
        return next_session_id.fetch_add(1, std::memory_order_relaxed);
    }

    Session& get_or_create_session(uint64_t sid, const std::string& model_id,
                                   uint64_t family_id) {
        std::lock_guard lock(sessions_mutex);
        auto it = sessions.find(sid);
        if (it != sessions.end()) return it->second;

        Session s;
        s.session_id = sid;
        s.family_id = family_id;
        s.model_id = model_id;
        s.state = SessionState::Active;
        s.last_active = std::chrono::steady_clock::now();
        auto [ins, ok] = sessions.emplace(sid, std::move(s));
        return ins->second;
    }
};

// ── Public interface ──────────────────────────────────────────────────────────

SessionManager::SessionManager(SessionManagerConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

SessionManager::~SessionManager() = default;

std::string SessionManager::complete(const api::ChatCompletionRequest& req,
                                     TokenCallback on_token) {
    // 1. Canonicalize prefix and compute family ID.
    const std::string canonical =
        extract_canonical_prefix(req.messages, impl_->config.prefix_opts);
    const uint64_t family_id =
        canonical.empty() ? 0 : make_family_id(req.model, canonical);

    // 2. Prefix cache lookup.
    bool cache_hit = false;
    if (family_id != 0) {
        cache_hit = impl_->prefix_store.contains(family_id);
    }

    auto& metrics = Metrics::instance();
    if (cache_hit) {
        metrics.record_prefix_hit();
    } else {
        metrics.record_prefix_miss();
        // In a real implementation: run prefill through llama.cpp, store snapshot.
        if (family_id != 0) {
            auto snapshot = std::make_shared<KVSnapshot>();
            snapshot->family_id = family_id;
            snapshot->model_id = req.model;
            snapshot->token_count = static_cast<uint32_t>(canonical.size() / 4);
            impl_->prefix_store.insert(std::move(snapshot));
        }
    }

    // 3. Determine session ID.
    uint64_t sid = impl_->allocate_session_id();
    if (req.session_id.has_value()) {
        try {
            sid = std::stoull(*req.session_id);
            metrics.record_session_resumed();
        } catch (...) {
            sid = impl_->allocate_session_id();
        }
    } else {
        metrics.record_session_created();
    }

    // 4. Register session.
    auto& session = impl_->get_or_create_session(sid, req.model, family_id);
    session.history = req.messages;
    session.last_active = std::chrono::steady_clock::now();

    // 5. Build a prompt summary for the stub backend.
    std::string prompt_summary;
    for (const auto& m : req.messages) prompt_summary += m.role + ":" + m.content + " ";

    // 6. Run inference (stub or real backend).
    return backend::generate_stub(prompt_summary, req.max_tokens, on_token);
}

bool SessionManager::pause(uint64_t session_id) {
    std::lock_guard lock(impl_->sessions_mutex);
    auto it = impl_->sessions.find(session_id);
    if (it == impl_->sessions.end()) return false;
    it->second.state = SessionState::Paused;
    return true;
}

bool SessionManager::resume(uint64_t session_id) {
    std::lock_guard lock(impl_->sessions_mutex);
    auto it = impl_->sessions.find(session_id);
    if (it == impl_->sessions.end()) return false;
    if (it->second.state != SessionState::Paused) return false;
    it->second.state = SessionState::Active;
    it->second.last_active = std::chrono::steady_clock::now();
    Metrics::instance().record_session_resumed();
    return true;
}

SessionManager::Stats SessionManager::stats() const {
    Stats s;
    {
        std::lock_guard lock(impl_->sessions_mutex);
        s.active_sessions = std::count_if(
            impl_->sessions.begin(), impl_->sessions.end(),
            [](const auto& kv) { return kv.second.state == SessionState::Active; });
    }
    s.prefix_hits = Metrics::instance().prefix_hits();
    s.prefix_misses = Metrics::instance().prefix_misses();
    s.fallback_events = Metrics::instance().fallbacks();
    s.prefix_store_bytes = impl_->prefix_store.total_bytes();
    return s;
}

}  // namespace kv3d
