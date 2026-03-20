#include "kv3d/kv/delta_codec.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace kv3d {

// ── Encode ────────────────────────────────────────────────────────────────────

DeltaCodec::EncodeResult DeltaCodec::encode(std::span<const float> base,
                                            std::span<const float> full) const {
    if (base.size() != full.size()) {
        throw std::invalid_argument("DeltaCodec::encode: base and full must have equal length");
    }

    const size_t n = base.size();
    EncodeResult result;
    result.data.resize(n);

    if (n == 0) {
        result.scale = 1.0f;
        return result;
    }

    // Compute deltas and find the absolute maximum for scale computation.
    std::vector<float> deltas(n);
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        deltas[i] = full[i] - base[i];
        max_abs = std::max(max_abs, std::abs(deltas[i]));
    }

    // Scale: map [-max_abs, +max_abs] -> [-127, 127].
    // If all deltas are zero (perfect prefix match), scale stays 1.0.
    result.scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    const float inv_scale = 1.0f / result.scale;

    for (size_t i = 0; i < n; ++i) {
        float q = deltas[i] * inv_scale;
        // Round-to-nearest, clamp to [-127, 127] (leave -128 as sentinel).
        int32_t qi = static_cast<int32_t>(std::round(q));
        qi = std::clamp(qi, -127, 127);
        result.data[i] = static_cast<int8_t>(qi);
    }

    return result;
}

// ── Decode ────────────────────────────────────────────────────────────────────

std::vector<float> DeltaCodec::decode(std::span<const float> base,
                                      std::span<const int8_t> delta,
                                      float scale) const {
    if (base.size() != delta.size()) {
        throw std::invalid_argument("DeltaCodec::decode: base and delta must have equal length");
    }

    const size_t n = base.size();
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = base[i] + static_cast<float>(delta[i]) * scale;
    }
    return out;
}

// ── Block-level helpers ───────────────────────────────────────────────────────

SessionDelta DeltaCodec::encode_block(uint64_t session_id,
                                      const KVBlock& base_block,
                                      const KVBlock& full_block) const {
    // Encode keys
    auto [key_data, key_scale] = encode(base_block.keys, full_block.keys);
    // Encode values (use same scale for simplicity; could be per-tensor)
    auto [val_data, val_scale] = encode(base_block.values, full_block.values);

    // Use the larger scale to avoid clipping on either tensor.
    const float scale = std::max(key_scale, val_scale);

    // Re-encode with the unified scale if it differs.
    SessionDelta delta;
    delta.session_id = session_id;
    delta.family_id = 0;  // caller sets this
    delta.delta_token_count = full_block.token_count - base_block.token_count;
    delta.scale_factor = scale;

    if (scale == key_scale) {
        delta.compressed_keys = std::move(key_data);
    } else {
        auto [kd, _] = encode(base_block.keys, full_block.keys);
        delta.compressed_keys = std::move(kd);
    }
    delta.compressed_values = std::move(val_data);
    return delta;
}

KVBlock DeltaCodec::decode_block(const KVBlock& base_block,
                                 const SessionDelta& delta) const {
    KVBlock out;
    out.layer_idx = base_block.layer_idx;
    out.token_offset = base_block.token_offset;
    out.token_count = base_block.token_count + delta.delta_token_count;
    out.keys = decode(base_block.keys, delta.compressed_keys, delta.scale_factor);
    out.values = decode(base_block.values, delta.compressed_values, delta.scale_factor);
    return out;
}

}  // namespace kv3d
