#include "kv3d/kv/delta_codec.hpp"
#include "kv3d/kv/kv_block.hpp"
#include "kv3d/kv/prefix_store.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numeric>
#include <vector>

using namespace kv3d;
using Catch::Matchers::WithinAbs;

// ── DeltaCodec ────────────────────────────────────────────────────────────────

static std::vector<float> linspace(float start, float end, size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i)
        v[i] = start + (end - start) * static_cast<float>(i) / static_cast<float>(n - 1);
    return v;
}

TEST_CASE("DeltaCodec: encode then decode round-trips within quantization error",
          "[delta_codec]") {
    DeltaCodec codec;

    const size_t N = 256;
    auto base = linspace(-1.0f, 1.0f, N);
    auto full = linspace(-0.8f, 1.2f, N);  // small shift

    const auto [encoded, scale] = codec.encode(base, full);
    REQUIRE(encoded.size() == N);
    REQUIRE(scale > 0.0f);

    const auto decoded = codec.decode(base, encoded, scale);
    REQUIRE(decoded.size() == N);

    // Max quantization error = scale (1 int8 step)
    for (size_t i = 0; i < N; ++i) {
        REQUIRE_THAT(decoded[i] - full[i], WithinAbs(0.0, static_cast<double>(scale)));
    }
}

TEST_CASE("DeltaCodec: zero delta encodes to all-zeros", "[delta_codec]") {
    DeltaCodec codec;
    const std::vector<float> base(64, 1.5f);
    const std::vector<float> full(64, 1.5f);  // identical

    const auto [encoded, scale] = codec.encode(base, full);
    for (auto v : encoded) REQUIRE(v == 0);
}

TEST_CASE("DeltaCodec: compression ratio is 4x", "[delta_codec]") {
    REQUIRE(DeltaCodec::compression_ratio() == 4.0f);
}

TEST_CASE("DeltaCodec: encode rejects mismatched sizes", "[delta_codec]") {
    DeltaCodec codec;
    std::vector<float> a(10, 0.0f), b(20, 0.0f);
    REQUIRE_THROWS_AS(codec.encode(a, b), std::invalid_argument);
}

TEST_CASE("DeltaCodec: large delta is clamped to int8 range", "[delta_codec]") {
    DeltaCodec codec;
    const std::vector<float> base(8, 0.0f);
    const std::vector<float> full(8, 1000.0f);  // huge delta

    const auto [encoded, scale] = codec.encode(base, full);
    for (auto v : encoded) {
        REQUIRE(v >= -127);
        REQUIRE(v <= 127);
    }
}

// ── PrefixStore ───────────────────────────────────────────────────────────────

TEST_CASE("PrefixStore: insert and retrieve", "[prefix_store]") {
    PrefixStore store;

    auto snap = std::make_shared<KVSnapshot>();
    snap->family_id = 42;
    snap->token_count = 10;

    store.insert(snap);
    REQUIRE(store.contains(42));
    REQUIRE(store.size() == 1);

    auto retrieved = store.get(42);
    REQUIRE(retrieved != nullptr);
    REQUIRE(retrieved->family_id == 42);
}

TEST_CASE("PrefixStore: miss returns nullptr", "[prefix_store]") {
    PrefixStore store;
    REQUIRE(store.get(999) == nullptr);
    REQUIRE(!store.contains(999));
}

TEST_CASE("PrefixStore: evict removes entry", "[prefix_store]") {
    PrefixStore store;

    auto snap = std::make_shared<KVSnapshot>();
    snap->family_id = 7;
    store.insert(snap);
    REQUIRE(store.contains(7));

    store.evict(7);
    REQUIRE(!store.contains(7));
    REQUIRE(store.size() == 0);
}

TEST_CASE("PrefixStore: evict_lru removes oldest entry", "[prefix_store]") {
    PrefixStore store;

    for (uint64_t id = 1; id <= 3; ++id) {
        auto s = std::make_shared<KVSnapshot>();
        s->family_id = id;
        store.insert(s);
    }
    REQUIRE(store.size() == 3);

    store.evict_lru();
    REQUIRE(store.size() == 2);
    // family_id 1 was inserted first so it should be the LRU
    REQUIRE(!store.contains(1));
}

TEST_CASE("PrefixStore: total_bytes accounts for all snapshots", "[prefix_store]") {
    PrefixStore store;

    for (uint64_t id = 0; id < 4; ++id) {
        auto s = std::make_shared<KVSnapshot>();
        s->family_id = id;
        KVBlock blk;
        blk.keys.assign(100, 0.0f);
        blk.values.assign(100, 0.0f);
        s->blocks.push_back(std::move(blk));
        store.insert(std::move(s));
    }

    // Each snapshot has 200 floats = 800 bytes; 4 snapshots = 3200 bytes
    REQUIRE(store.total_bytes() == 4 * 200 * sizeof(float));
}
