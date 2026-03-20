#include "kv3d/core/canonical_prompt.hpp"
#include "kv3d/core/prefix_hash.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace kv3d;

TEST_CASE("hash_prefix is deterministic", "[prefix_hash]") {
    REQUIRE(hash_prefix("hello world") == hash_prefix("hello world"));
    REQUIRE(hash_prefix("") == hash_prefix(""));
}

TEST_CASE("hash_prefix produces distinct values for distinct inputs", "[prefix_hash]") {
    REQUIRE(hash_prefix("system: you are a helpful assistant") !=
            hash_prefix("system: you are a coding assistant"));
    REQUIRE(hash_prefix("abc") != hash_prefix("bcd"));
    REQUIRE(hash_prefix("a") != hash_prefix("b"));
}

TEST_CASE("hash_prefix empty string", "[prefix_hash]") {
    const uint64_t h = hash_prefix("");
    REQUIRE(h == FNV_OFFSET_64);  // no bytes mixed in: result is the offset basis
}

TEST_CASE("combine_hashes is not commutative (order matters)", "[prefix_hash]") {
    const uint64_t h1 = hash_prefix("model-A");
    const uint64_t h2 = hash_prefix("prefix-text");
    // combine is intentionally order-sensitive so model+prefix != prefix+model
    REQUIRE(combine_hashes(h1, h2) != combine_hashes(h2, h1));
}

TEST_CASE("make_family_id varies by model and prefix", "[prefix_hash]") {
    const uint64_t a = make_family_id("qwen2.5-7b", "you are a helpful assistant");
    const uint64_t b = make_family_id("llama3-8b", "you are a helpful assistant");
    const uint64_t c = make_family_id("qwen2.5-7b", "you are a coding assistant");
    REQUIRE(a != b);
    REQUIRE(a != c);
    REQUIRE(b != c);
}

TEST_CASE("make_family_id is stable across calls", "[prefix_hash]") {
    const std::string model = "qwen2.5-7b-instruct";
    const std::string prefix = "<system>You are a helpful assistant.</system>";
    REQUIRE(make_family_id(model, prefix) == make_family_id(model, prefix));
}

// ── Canonical prompt tests ────────────────────────────────────────────────────

TEST_CASE("normalize_whitespace collapses runs", "[canonical]") {
    REQUIRE(normalize_whitespace("hello   world") == "hello world");
    REQUIRE(normalize_whitespace("  leading") == "leading");
    REQUIRE(normalize_whitespace("trailing  ") == "trailing");
    REQUIRE(normalize_whitespace("a\r\nb") == "a b");
    REQUIRE(normalize_whitespace("") == "");
}

TEST_CASE("prefix_message_count: no system message -> 0", "[canonical]") {
    std::vector<ChatMessage> msgs = {{"user", "Hello!"}};
    REQUIRE(prefix_message_count(msgs) == 0);
}

TEST_CASE("prefix_message_count: system only -> 1", "[canonical]") {
    std::vector<ChatMessage> msgs = {{"system", "You are helpful."},
                                     {"user", "Help me."}};
    PrefixExtractionOptions opts;
    opts.system_only = true;
    REQUIRE(prefix_message_count(msgs, opts) == 1);
}

TEST_CASE("prefix_message_count: system + first user -> 2", "[canonical]") {
    std::vector<ChatMessage> msgs = {{"system", "You are helpful."},
                                     {"user", "What is 2+2?"},
                                     {"assistant", "4"},
                                     {"user", "Thanks"}};
    REQUIRE(prefix_message_count(msgs) == 2);
}

TEST_CASE("extract_canonical_prefix: consistent format", "[canonical]") {
    std::vector<ChatMessage> msgs = {{"system", "  You are helpful.  "},
                                     {"user", "Hello"}};
    const std::string canon = extract_canonical_prefix(msgs);
    REQUIRE(canon == "<system>You are helpful.</system><user>Hello</user>");
}

TEST_CASE("extract_canonical_prefix: same content -> same hash", "[canonical]") {
    std::vector<ChatMessage> msgs1 = {{"system", "You are an assistant."},
                                      {"user", "Hi"}};
    std::vector<ChatMessage> msgs2 = {{"system", "You are an assistant."},
                                      {"user", "Hi"},
                                      {"assistant", "Hello! How can I help?"}};

    const std::string c1 = extract_canonical_prefix(msgs1);
    const std::string c2 = extract_canonical_prefix(msgs2);
    REQUIRE(c1 == c2);  // assistant turn is not part of the prefix
}

TEST_CASE("extract_canonical_prefix: empty for no system message", "[canonical]") {
    std::vector<ChatMessage> msgs = {{"user", "Hello"}};
    REQUIRE(extract_canonical_prefix(msgs).empty());
}
