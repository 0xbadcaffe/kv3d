#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "kv3d/kv/experimental/group_codec.hpp"

#include <cmath>
#include <random>
#include <thread>

using namespace kv3d;
using namespace kv3d::experimental;
using Catch::Matchers::WithinAbs;

// ── Test helpers ──────────────────────────────────────────────────────────────

static constexpr uint32_t N_HEADS   = 8;
static constexpr uint32_t HEAD_DIM  = 64;
static constexpr uint32_t N_TOKENS  = 16;
static constexpr size_t   BLOCK_DIM = N_HEADS * N_TOKENS * HEAD_DIM; // 8192

static KVBlock make_block(uint32_t layer, float fill_k, float fill_v,
                          uint32_t tokens = N_TOKENS) {
    KVBlock b;
    b.layer_idx    = layer;
    b.token_offset = 0;
    b.token_count  = tokens;
    b.keys.assign(N_HEADS * tokens * HEAD_DIM, fill_k);
    b.values.assign(N_HEADS * tokens * HEAD_DIM, fill_v);
    return b;
}

static KVBlock make_noisy(uint32_t layer, float fill_k, float fill_v,
                          float noise_sigma, uint64_t seed = 42) {
    KVBlock b = make_block(layer, fill_k, fill_v);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, noise_sigma);
    for (auto& v : b.keys)   v += dist(rng);
    for (auto& v : b.values) v += dist(rng);
    return b;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

TEST_CASE("GroupCodec: encode then decode round-trips within quantization error",
          "[GroupCodec]") {
    GroupCodec codec;
    const auto block   = make_block(0, 0.5f, -0.3f);
    const auto encoded = codec.encode(block);
    const auto decoded = codec.decode(block, encoded);

    REQUIRE(decoded.keys.size()   == block.keys.size());
    REQUIRE(decoded.values.size() == block.values.size());
    REQUIRE(decoded.layer_idx     == block.layer_idx);
    REQUIRE(decoded.token_count   == block.token_count);

    // Quantization error is bounded by scale/2 = max_abs / 254
    // For a first encode the block IS the prototype → residual = 0 → perfect.
    for (size_t i = 0; i < block.keys.size(); ++i)
        REQUIRE_THAT(decoded.keys[i], WithinAbs(block.keys[i], 0.01f));
    for (size_t i = 0; i < block.values.size(); ++i)
        REQUIRE_THAT(decoded.values[i], WithinAbs(block.values[i], 0.01f));
}

TEST_CASE("GroupCodec: first encode of a block IS the prototype → similarity 1",
          "[GroupCodec]") {
    GroupCodec codec;
    const auto res = codec.encode(make_block(0, 1.0f, 0.5f));
    REQUIRE(res.similarity == 1.0f);
    REQUIRE(codec.prototype_count() == 1);
}

TEST_CASE("GroupCodec: near-identical blocks share a prototype", "[GroupCodec]") {
    BlockGroupConfig cfg;
    cfg.similarity_threshold = 0.80f;
    GroupCodec codec(cfg);

    const auto b1 = make_noisy(0, 1.0f, 0.5f, 0.001f, 1);
    const auto b2 = make_noisy(0, 1.0f, 0.5f, 0.001f, 2);

    const auto r1 = codec.encode(b1);
    const auto r2 = codec.encode(b2);

    REQUIRE(r1.prototype_id == r2.prototype_id);
    REQUIRE(r2.similarity   >= 0.80f);
    REQUIRE(codec.prototype_count() == 1);
}

TEST_CASE("GroupCodec: dissimilar blocks get distinct prototypes", "[GroupCodec]") {
    BlockGroupConfig cfg;
    cfg.similarity_threshold = 0.80f;
    GroupCodec codec(cfg);

    // Orthogonal fill values: cosine sim between [1,1,...] and [-1,-1,...] = -1
    const auto b1 = make_block(0,  1.0f,  0.5f);
    const auto b2 = make_block(0, -1.0f, -0.5f);

    const auto r1 = codec.encode(b1);
    const auto r2 = codec.encode(b2);

    REQUIRE(r1.prototype_id != r2.prototype_id);
    REQUIRE(codec.prototype_count() == 2);
}

TEST_CASE("GroupCodec: compression ratio is 4x for a new (first-seen) block",
          "[GroupCodec]") {
    GroupCodec codec;
    // First encode: residual is all-zero → scale = 1 → int8 = 0 everywhere.
    // Ratio = (n * 4) / (n * 1 + 2 * 4) ≈ 4 for large n.
    const auto r = codec.encode(make_block(0, 0.7f, 0.2f));
    REQUIRE(r.compression_ratio() >= 3.9f);
}

TEST_CASE("GroupCodec: residual of similar block is smaller than raw delta",
          "[GroupCodec]") {
    BlockGroupConfig cfg;
    cfg.similarity_threshold = 0.80f;
    GroupCodec codec(cfg);

    const auto base = make_block(0, 1.0f, 0.5f);
    const auto noisy = make_noisy(0, 1.0f, 0.5f, 0.01f, 99);

    codec.encode(base);                       // seeds prototype
    const auto r = codec.encode(noisy);       // residual vs prototype

    // The prototype was seeded from `base`; `noisy` is base + small noise.
    // Residual max-abs should be ≈ noise_sigma, while the raw values are ≈ 1.0.
    REQUIRE(r.scale_keys   < 0.1f);   // scale = max_abs/127, expect < 0.1
    REQUIRE(r.scale_values < 0.1f);
}

TEST_CASE("GroupCodec: prototype pool respects max_prototypes cap", "[GroupCodec]") {
    BlockGroupConfig cfg;
    cfg.max_prototypes     = 4;
    cfg.similarity_threshold = 0.99f; // very strict → nearly every block is new
    GroupCodec codec(cfg);

    for (int i = 0; i < 12; ++i)
        std::ignore = codec.encode(make_block(0, static_cast<float>(i) * 10.0f, 0.0f));

    REQUIRE(codec.prototype_count() <= 4);
}

TEST_CASE("GroupCodec: blocks from different layers get separate prototypes",
          "[GroupCodec]") {
    GroupCodec codec;

    const auto r0 = codec.encode(make_block(0, 1.0f, 0.5f));
    const auto r1 = codec.encode(make_block(1, 1.0f, 0.5f)); // same content, different layer

    // Must not reuse across layers
    REQUIRE(r0.prototype_id != r1.prototype_id);
    REQUIRE(codec.prototype_count() == 2);
}

TEST_CASE("GroupCodec: statistics are updated after encodes", "[GroupCodec]") {
    GroupCodec codec;
    REQUIRE(codec.total_encodes() == 0);
    REQUIRE(codec.mean_similarity() == 0.0);

    for (int i = 0; i < 5; ++i)
        codec.encode(make_block(0, static_cast<float>(i), 0.0f));

    REQUIRE(codec.total_encodes()        == 5);
    REQUIRE(codec.mean_similarity()       > 0.0);
    REQUIRE(codec.mean_compression_ratio() >= 1.0);
    REQUIRE(codec.total_prototype_bytes()  > 0);
}

TEST_CASE("GroupCodec: clear resets pool and statistics", "[GroupCodec]") {
    GroupCodec codec;
    codec.encode(make_block(0, 1.0f, 0.0f));
    REQUIRE(codec.prototype_count() == 1);

    codec.clear();
    REQUIRE(codec.prototype_count()        == 0);
    REQUIRE(codec.total_encodes()          == 0);
    REQUIRE(codec.mean_similarity()        == 0.0);
    REQUIRE(codec.mean_compression_ratio() == 0.0);
}

TEST_CASE("GroupCodec: evict_lru shrinks pool", "[GroupCodec]") {
    BlockGroupConfig cfg;
    cfg.similarity_threshold = 0.99f; // force many prototypes
    cfg.min_assignments      = 1;
    GroupCodec codec(cfg);

    for (int i = 0; i < 4; ++i)
        std::ignore = codec.encode(make_block(0, static_cast<float>(i) * 10.0f, 0.0f));

    const size_t before = codec.prototype_count();
    codec.evict_lru();
    REQUIRE(codec.prototype_count() == before - 1);
}

TEST_CASE("GroupCodec: round-trip with noisy block within 2× quantization error",
          "[GroupCodec]") {
    BlockGroupConfig cfg;
    cfg.similarity_threshold = 0.70f;
    GroupCodec codec(cfg);

    const auto base  = make_block(0, 0.8f, 0.3f);
    const auto noisy = make_noisy(0, 0.8f, 0.3f, 0.05f, 7);

    std::ignore = codec.encode(base);
    const auto encoded = codec.encode(noisy);
    const auto decoded = codec.decode(noisy, encoded);

    // Quantization error ≤ scale/2; scale ≤ max_residual/127 ≈ noise_sigma * 3 / 127
    const float tol = 0.05f * 3.0f / 127.0f * 2.0f + 1e-3f; // generous
    for (size_t i = 0; i < noisy.keys.size(); ++i)
        REQUIRE_THAT(decoded.keys[i], WithinAbs(noisy.keys[i], tol));
}

TEST_CASE("GroupCodec: thread-safety smoke test", "[GroupCodec][.slow]") {
    GroupCodec codec;
    constexpr int N_THREADS = 8;
    constexpr int OPS_EACH  = 50;

    std::vector<std::thread> threads;
    threads.reserve(N_THREADS);
    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&codec, t] {
            std::mt19937 rng(static_cast<unsigned>(t));
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (int op = 0; op < OPS_EACH; ++op) {
                const auto block = make_block(0, dist(rng), dist(rng));
                const auto enc   = codec.encode(block);
                (void)codec.decode(block, enc);
            }
        });
    }
    for (auto& th : threads) th.join();

    REQUIRE(codec.total_encodes() == static_cast<uint64_t>(N_THREADS * OPS_EACH));
}
