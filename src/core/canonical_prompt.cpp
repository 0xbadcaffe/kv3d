#include "kv3d/core/canonical_prompt.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace kv3d {

std::string normalize_whitespace(std::string_view input) noexcept {
    std::string out;
    out.reserve(input.size());

    bool in_space = true;  // treat leading spaces as already seen

    for (char c : input) {
        // Normalize \r\n and \r -> \n
        if (c == '\r') continue;

        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!in_space && !out.empty()) {
                // Collapse run to a single space
                out.push_back(' ');
                in_space = true;
            }
        } else {
            out.push_back(c);
            in_space = false;
        }
    }

    // Strip trailing space that was appended
    if (!out.empty() && out.back() == ' ') out.pop_back();

    return out;
}

size_t prefix_message_count(const std::vector<ChatMessage>& messages,
                             const PrefixExtractionOptions& opts) noexcept {
    if (messages.empty()) return 0;

    // Always include a leading system message if present.
    size_t count = 0;
    if (messages[0].role == "system") {
        count = 1;
        if (opts.system_only) return count;
    }

    if (count == 0) {
        // No system message: no prefix to share under default policy.
        return 0;
    }

    // Optionally extend the prefix to include the first user turn.
    if (!opts.system_only && count < messages.size() && messages[count].role == "user") {
        count += 1;
    }

    return count;
}

std::string extract_canonical_prefix(const std::vector<ChatMessage>& messages,
                                     const PrefixExtractionOptions& opts) {
    const size_t n = prefix_message_count(messages, opts);
    if (n == 0) return {};

    std::ostringstream out;
    for (size_t i = 0; i < n; ++i) {
        const auto& msg = messages[i];
        out << '<' << msg.role << '>';
        if (opts.normalize_whitespace) {
            out << normalize_whitespace(msg.content);
        } else {
            out << msg.content;
        }
        out << "</" << msg.role << '>';
    }
    return out.str();
}

}  // namespace kv3d
