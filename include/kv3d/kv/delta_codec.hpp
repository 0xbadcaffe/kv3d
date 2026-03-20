#pragma once

#include "kv3d/kv/kv_block.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace kv3d {

/// Quantization-based KV delta codec.
///
/// Encodes the difference between a full session KV block and the shared prefix
/// KV block using symmetric int8 quantization:
///
///   delta[i] = full[i] - base[i]
///   q[i]     = clamp(round(delta[i] / scale), -127, 127)
///   scale    = max(|delta|) / 127
///
/// This achieves ~4x compression (fp32 -> int8) at the cost of small, bounded
/// quantization error.  The auto-fallback guardrail monitors quality drift.
class DeltaCodec {
public:
    DeltaCodec() = default;

    /// Encode (full - base) into int8 with a per-block scale.
    /// base and full must have equal length.
    struct EncodeResult {
        std::vector<int8_t> data;
        float scale;
    };
    [[nodiscard]] EncodeResult encode(std::span<const float> base,
                                      std::span<const float> full) const;

    /// Decode: reconstruct full ≈ base + dequantize(delta, scale).
    [[nodiscard]] std::vector<float> decode(std::span<const float> base,
                                            std::span<const int8_t> delta,
                                            float scale) const;

    /// Encode a whole KVBlock pair (session block vs prefix block).
    [[nodiscard]] SessionDelta encode_block(uint64_t session_id,
                                            const KVBlock& base_block,
                                            const KVBlock& full_block) const;

    /// Reconstruct a full KVBlock from prefix block + delta.
    [[nodiscard]] KVBlock decode_block(const KVBlock& base_block,
                                       const SessionDelta& delta) const;

    /// Compression ratio: input bytes / output bytes (higher is better).
    [[nodiscard]] static float compression_ratio() noexcept {
        return sizeof(float) / sizeof(int8_t);  // 4×
    }
};

}  // namespace kv3d
