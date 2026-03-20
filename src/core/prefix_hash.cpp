#include "kv3d/core/prefix_hash.hpp"

namespace kv3d {

uint64_t hash_prefix(std::string_view data) noexcept {
    uint64_t h = FNV_OFFSET_64;
    for (unsigned char c : data) {
        h ^= static_cast<uint64_t>(c);
        h *= FNV_PRIME_64;
    }
    return h;
}

uint64_t combine_hashes(uint64_t h1, uint64_t h2) noexcept {
    // Avalanche mix: XOR then multiply by the prime to propagate all bits.
    h1 ^= h2;
    h1 *= FNV_PRIME_64;
    // Extra rotation for better distribution when h2 has low entropy.
    h1 ^= (h1 >> 33);
    h1 *= 0xff51afd7ed558ccdULL;
    h1 ^= (h1 >> 33);
    return h1;
}

uint64_t make_family_id(std::string_view model_id,
                        std::string_view canonical_prefix) noexcept {
    return combine_hashes(hash_prefix(model_id), hash_prefix(canonical_prefix));
}

}  // namespace kv3d
