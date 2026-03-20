#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace kv3d {

struct ChatMessage {
    std::string role;     // "system" | "user" | "assistant" | "tool"
    std::string content;
};

/// Controls what portion of the message list is treated as the shared prefix.
struct PrefixExtractionOptions {
    /// Include only the system message (role == "system") in the prefix.
    /// When false, also include the first user turn.
    bool system_only = false;
    /// Normalize whitespace (collapse runs, strip edges) before hashing.
    bool normalize_whitespace = true;
};

/// Canonicalize and extract the prefix string that will be hashed.
/// Returns an empty string when no shareable prefix can be determined.
[[nodiscard]] std::string extract_canonical_prefix(
    const std::vector<ChatMessage>& messages,
    const PrefixExtractionOptions& opts = {});

/// Count how many leading messages belong to the prefix.
[[nodiscard]] size_t prefix_message_count(const std::vector<ChatMessage>& messages,
                                          const PrefixExtractionOptions& opts = {}) noexcept;

/// Normalize whitespace in a string: collapse internal runs to a single space,
/// strip leading/trailing whitespace, normalize \r\n -> \n.
[[nodiscard]] std::string normalize_whitespace(std::string_view input) noexcept;

}  // namespace kv3d
