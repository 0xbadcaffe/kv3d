#include "kv3d/kv/experimental/group_codec.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>

namespace kv3d::experimental {

// ── GroupEncodeResult ─────────────────────────────────────────────────────────

float GroupEncodeResult::compression_ratio() const noexcept {
    const size_t n = residual_keys.size() + residual_values.size();
    if (n == 0) return 1.0f;
    const size_t raw_bytes = n * sizeof(float);
    const size_t cmp_bytes = n * sizeof(int8_t) + 2 * sizeof(float); // int8 + 2 scales
    return static_cast<float>(raw_bytes) / static_cast<float>(cmp_bytes);
}

// ── Internal prototype representation ────────────────────────────────────────

struct Prototype {
    uint64_t             id;
    uint32_t             layer_idx;
    uint32_t             token_count;
    std::vector<float>   keys;               // running centroid
    std::vector<float>   values;
    uint64_t             assignment_count{0};
    uint64_t             last_access_epoch{0};
};

// ── File-local helpers ────────────────────────────────────────────────────────

namespace {

// Symmetric int8 quantization of a float residual vector.
// Returns quantized bytes and sets out_scale = max_abs / 127.
std::vector<int8_t> quantize_residual(std::span<const float> deltas,
                                      float& out_scale) noexcept {
    if (deltas.empty()) {
        out_scale = 1.0f;
        return {};
    }
    float max_abs = 0.0f;
    for (float d : deltas) max_abs = std::max(max_abs, std::abs(d));

    out_scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    const float inv = 1.0f / out_scale;

    std::vector<int8_t> out(deltas.size());
    for (size_t i = 0; i < deltas.size(); ++i) {
        const int32_t q = static_cast<int32_t>(std::round(deltas[i] * inv));
        out[i] = static_cast<int8_t>(std::clamp(q, -127, 127));
    }
    return out;
}

std::vector<float> dequantize_residual(std::span<const int8_t> data,
                                       float scale) noexcept {
    std::vector<float> out(data.size());
    for (size_t i = 0; i < data.size(); ++i)
        out[i] = static_cast<float>(data[i]) * scale;
    return out;
}

// Cosine similarity over a subsampled index set.
// Returns 0 if norms are zero or index set is empty.
float cosine_similarity_probed(std::span<const float>    a,
                                std::span<const float>    b,
                                std::span<const uint32_t> indices) noexcept {
    if (indices.empty() || a.size() != b.size()) return 0.0f;

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (uint32_t idx : indices) {
        if (idx >= a.size()) break;
        const double ai = a[idx], bi = b[idx];
        dot += ai * bi;
        na  += ai * ai;
        nb  += bi * bi;
    }
    const double denom = std::sqrt(na * nb);
    return (denom > 0.0) ? static_cast<float>(dot / denom) : 0.0f;
}

} // anonymous namespace

// ── GroupCodec::Impl ──────────────────────────────────────────────────────────

struct GroupCodec::Impl {
    BlockGroupConfig       cfg;
    mutable std::mutex     mu;

    std::vector<Prototype>  pool;
    std::vector<uint32_t>   probe_indices; // stable subsampled key dims

    uint64_t epoch        = 0;
    uint64_t next_proto_id = 0;

    // Rolling stats (updated under mu)
    uint64_t total_encodes_      = 0;
    double   sum_similarity_     = 0.0;
    double   sum_compression_    = 0.0;

    // Build probe_indices for blocks whose key vector has `key_dim` elements.
    // Called lazily on first encode (or when shape changes).
    void init_probes(size_t key_dim) {
        if (!probe_indices.empty()) return;

        const size_t n = std::min(static_cast<size_t>(cfg.probe_dims), key_dim);
        probe_indices.resize(n);

        // Evenly-strided sample — deterministic, no RNG needed.
        const double stride = static_cast<double>(key_dim) / static_cast<double>(n);
        for (size_t i = 0; i < n; ++i)
            probe_indices[i] = static_cast<uint32_t>(
                std::min(static_cast<size_t>(std::floor(i * stride)), key_dim - 1));

        // Deduplicate (can occur when key_dim < probe_dims)
        probe_indices.erase(
            std::unique(probe_indices.begin(), probe_indices.end()),
            probe_indices.end());
    }

    // Find the prototype with highest probed cosine similarity.
    // Returns {pool_index, similarity}; pool_index == npos if pool is empty.
    std::pair<size_t, float> find_nearest(const KVBlock& block) const {
        float  best_sim = -2.0f;
        size_t best_idx = std::numeric_limits<size_t>::max();

        for (size_t i = 0; i < pool.size(); ++i) {
            const auto& p = pool[i];
            if (p.layer_idx   != block.layer_idx   ||
                p.keys.size() != block.keys.size())   continue;

            const float sim = cosine_similarity_probed(p.keys, block.keys, probe_indices);
            if (sim > best_sim) {
                best_sim = sim;
                best_idx = i;
            }
        }
        return {best_idx, best_sim};
    }

    // EMA update of prototype centroid toward new_block.
    void update_ema(Prototype& p, const KVBlock& block) const {
        const float alpha     = cfg.ema_alpha;
        const float one_minus = 1.0f - alpha;
        for (size_t i = 0; i < p.keys.size();   ++i)
            p.keys[i]   = one_minus * p.keys[i]   + alpha * block.keys[i];
        for (size_t i = 0; i < p.values.size(); ++i)
            p.values[i] = one_minus * p.values[i] + alpha * block.values[i];
        ++p.assignment_count;
        p.last_access_epoch = epoch;
    }

    // Evict the LRU prototype that satisfies min_assignments.
    // Falls back to evicting the absolute LRU if none qualify.
    bool do_evict_lru() {
        if (pool.empty()) return false;

        auto score = [&](const Prototype& p) -> uint64_t {
            return (p.assignment_count >= cfg.min_assignments)
                   ? p.last_access_epoch
                   : (p.last_access_epoch + std::numeric_limits<uint32_t>::max()); // prefer deleting under-used
        };

        size_t lru_idx = 0;
        uint64_t lru_score = score(pool[0]);
        for (size_t i = 1; i < pool.size(); ++i) {
            if (score(pool[i]) < lru_score) {
                lru_score = score(pool[i]);
                lru_idx   = i;
            }
        }
        pool.erase(pool.begin() + static_cast<ptrdiff_t>(lru_idx));
        return true;
    }
};

// ── GroupCodec public interface ───────────────────────────────────────────────

GroupCodec::GroupCodec(BlockGroupConfig cfg)
    : impl_(std::make_unique<Impl>()) {
    impl_->cfg = std::move(cfg);
}

GroupCodec::~GroupCodec() = default;

GroupEncodeResult GroupCodec::encode(const KVBlock& block) {
    std::lock_guard lock(impl_->mu);

    impl_->init_probes(block.keys.size());
    ++impl_->epoch;

    auto [best_idx, best_sim] = impl_->find_nearest(block);

    const bool reuse = (best_idx != std::numeric_limits<size_t>::max())
                    && (best_sim >= impl_->cfg.similarity_threshold);

    if (!reuse) {
        // Need a new prototype — evict LRU first if at capacity.
        if (impl_->pool.size() >= impl_->cfg.max_prototypes)
            impl_->do_evict_lru();

        Prototype p;
        p.id                 = impl_->next_proto_id++;
        p.layer_idx          = block.layer_idx;
        p.token_count        = block.token_count;
        p.keys               = block.keys;
        p.values             = block.values;
        p.assignment_count   = 1;
        p.last_access_epoch  = impl_->epoch;
        best_idx             = impl_->pool.size();
        best_sim             = 1.0f;          // perfect match — it IS the prototype
        impl_->pool.push_back(std::move(p));
    } else {
        impl_->update_ema(impl_->pool[best_idx], block);
    }

    const Prototype& proto = impl_->pool[best_idx];

    // Compute residual = block - prototype (per-tensor)
    const size_t nk = block.keys.size(), nv = block.values.size();
    std::vector<float> dk(nk), dv(nv);
    for (size_t i = 0; i < nk; ++i) dk[i] = block.keys[i]   - proto.keys[i];
    for (size_t i = 0; i < nv; ++i) dv[i] = block.values[i] - proto.values[i];

    GroupEncodeResult res;
    res.prototype_id     = proto.id;
    res.similarity       = best_sim;
    res.residual_keys    = quantize_residual(dk, res.scale_keys);
    res.residual_values  = quantize_residual(dv, res.scale_values);

    impl_->total_encodes_++;
    impl_->sum_similarity_  += best_sim;
    impl_->sum_compression_ += res.compression_ratio();

    return res;
}

KVBlock GroupCodec::decode(const KVBlock& shape, const GroupEncodeResult& result) const {
    // Take a snapshot of the prototype to avoid holding the lock during
    // the (potentially large) vector reconstruction.
    std::vector<float> proto_keys, proto_values;
    {
        std::lock_guard lock(impl_->mu);
        const Prototype* p = nullptr;
        for (const auto& proto : impl_->pool)
            if (proto.id == result.prototype_id) { p = &proto; break; }
        if (!p)
            throw std::runtime_error("GroupCodec::decode: unknown prototype_id "
                                     + std::to_string(result.prototype_id));
        proto_keys   = p->keys;
        proto_values = p->values;
    }

    auto dk = dequantize_residual(result.residual_keys,   result.scale_keys);
    auto dv = dequantize_residual(result.residual_values, result.scale_values);

    KVBlock out;
    out.layer_idx    = shape.layer_idx;
    out.token_offset = shape.token_offset;
    out.token_count  = shape.token_count;
    out.keys.resize(proto_keys.size());
    out.values.resize(proto_values.size());

    for (size_t i = 0; i < out.keys.size();   ++i) out.keys[i]   = proto_keys[i]   + dk[i];
    for (size_t i = 0; i < out.values.size(); ++i) out.values[i] = proto_values[i] + dv[i];
    return out;
}

size_t GroupCodec::prototype_count() const {
    std::lock_guard lock(impl_->mu);
    return impl_->pool.size();
}

size_t GroupCodec::total_prototype_bytes() const {
    std::lock_guard lock(impl_->mu);
    size_t total = 0;
    for (const auto& p : impl_->pool)
        total += (p.keys.size() + p.values.size()) * sizeof(float);
    return total;
}

double GroupCodec::mean_similarity() const {
    std::lock_guard lock(impl_->mu);
    return impl_->total_encodes_ > 0
           ? impl_->sum_similarity_ / static_cast<double>(impl_->total_encodes_)
           : 0.0;
}

double GroupCodec::mean_compression_ratio() const {
    std::lock_guard lock(impl_->mu);
    return impl_->total_encodes_ > 0
           ? impl_->sum_compression_ / static_cast<double>(impl_->total_encodes_)
           : 0.0;
}

uint64_t GroupCodec::total_encodes() const {
    std::lock_guard lock(impl_->mu);
    return impl_->total_encodes_;
}

void GroupCodec::clear() {
    std::lock_guard lock(impl_->mu);
    impl_->pool.clear();
    impl_->probe_indices.clear();
    impl_->epoch           = 0;
    impl_->next_proto_id   = 0;
    impl_->total_encodes_  = 0;
    impl_->sum_similarity_ = 0.0;
    impl_->sum_compression_= 0.0;
}

bool GroupCodec::evict_lru() {
    std::lock_guard lock(impl_->mu);
    return impl_->do_evict_lru();
}

BlockGroupConfig GroupCodec::config() const {
    std::lock_guard lock(impl_->mu);
    return impl_->cfg;
}

void GroupCodec::set_config(BlockGroupConfig cfg) {
    std::lock_guard lock(impl_->mu);
    impl_->cfg = std::move(cfg);
}

} // namespace kv3d::experimental
