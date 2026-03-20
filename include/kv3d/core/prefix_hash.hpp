#pragma once

#include <cstdint>
#include <string_view>

namespace kv3d {

/// FNV-1a 64-bit constants.
inline constexpr uint64_t FNV_PRIME_64  = 0x00000100000001B3ULL;
inline constexpr uint64_t FNV_OFFSET_64 = 0xcbf29ce484222325ULL;

/// Hash a byte sequence with FNV-1a 64-bit.
[[nodiscard]] uint64_t hash_prefix(std::string_view data) noexcept;

/// Mix two hashes — commutative-safe combination.
[[nodiscard]] uint64_t combine_hashes(uint64_t h1, uint64_t h2) noexcept;

/// Build a family ID that is unique per (model, canonical_prefix) pair.
/// Sessions sharing both will reuse the same KV snapshot.
[[nodiscard]] uint64_t make_family_id(std::string_view model_id,
                                      std::string_view canonical_prefix) noexcept;

}  // namespace kv3d
