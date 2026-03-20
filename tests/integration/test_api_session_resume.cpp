#include "kv3d/api/types.hpp"
#include "kv3d/sched/session_manager.hpp"

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <string>
#include <vector>

using namespace kv3d;
using namespace kv3d::api;

// ── Helpers ───────────────────────────────────────────────────────────────────

static ChatCompletionRequest make_request(std::string system_prompt,
                                          std::string user_message,
                                          bool stream = false) {
    ChatCompletionRequest req;
    req.model = "stub-model";
    req.max_tokens = 32;
    req.stream = stream;
    req.messages = {{"system", std::move(system_prompt)}, {"user", std::move(user_message)}};
    return req;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_CASE("SessionManager: basic completion returns non-empty response",
          "[session_manager][integration]") {
    SessionManager mgr;
    auto req = make_request("You are a helpful assistant.", "What is the capital of France?");
    const std::string resp = mgr.complete(req);
    REQUIRE(!resp.empty());
}

TEST_CASE("SessionManager: shared prefix causes cache hit on second request",
          "[session_manager][integration]") {
    SessionManager mgr;

    const std::string shared_system = "You are a helpful assistant.";

    auto req1 = make_request(shared_system, "Hello!");
    mgr.complete(req1);

    const auto stats_before = mgr.stats();

    auto req2 = make_request(shared_system, "How are you?");
    mgr.complete(req2);

    const auto stats_after = mgr.stats();

    // Second request should have registered a prefix hit (same family_id).
    REQUIRE(stats_after.prefix_hits > stats_before.prefix_hits);
}

TEST_CASE("SessionManager: different system prompts yield different families",
          "[session_manager][integration]") {
    SessionManager mgr;

    mgr.complete(make_request("You are a coding assistant.", "Write a sort."));
    const uint64_t hits_a = mgr.stats().prefix_hits;

    mgr.complete(make_request("You are a creative writer.", "Write a poem."));
    // No shared prefix between two different system prompts — still misses only
    const uint64_t hits_b = mgr.stats().prefix_hits;

    // Both should hit after two calls to the same system prompt
    mgr.complete(make_request("You are a coding assistant.", "Write a search."));
    REQUIRE(mgr.stats().prefix_hits > hits_b);
}

TEST_CASE("SessionManager: streaming delivers tokens via callback",
          "[session_manager][integration]") {
    SessionManager mgr;

    std::atomic<int> token_count{0};
    std::string assembled;
    bool done_seen = false;

    auto req = make_request("You are helpful.", "Count to three.", /*stream=*/true);
    mgr.complete(req, [&](std::string_view tok, bool done) {
        assembled += tok;
        token_count.fetch_add(1, std::memory_order_relaxed);
        if (done) done_seen = true;
    });

    REQUIRE(token_count.load() > 0);
    REQUIRE(!assembled.empty());
    REQUIRE(done_seen);
}

TEST_CASE("SessionManager: pause and resume changes session state",
          "[session_manager][integration]") {
    SessionManager mgr;

    // Create a session by running a completion
    auto req = make_request("Be concise.", "What is 1+1?");
    mgr.complete(req);

    // Session ID 1 should have been created (first allocation)
    const bool paused = mgr.pause(1);
    REQUIRE(paused);

    const bool resumed = mgr.resume(1);
    REQUIRE(resumed);

    // Re-pausing an active session should work
    REQUIRE(mgr.pause(1));
}

TEST_CASE("SessionManager: stats are monotonically increasing", "[session_manager][integration]") {
    SessionManager mgr;

    const auto s0 = mgr.stats();

    for (int i = 0; i < 5; ++i) {
        mgr.complete(make_request("Be helpful.", "Query " + std::to_string(i)));
    }

    const auto s1 = mgr.stats();
    REQUIRE(s1.prefix_hits + s1.prefix_misses > s0.prefix_hits + s0.prefix_misses);
}
