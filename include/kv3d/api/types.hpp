#pragma once

#include "kv3d/core/canonical_prompt.hpp"

#include <optional>
#include <string>
#include <vector>

namespace kv3d::api {

// ── OpenAI-compatible request / response types ────────────────────────────────

struct ChatCompletionRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    float temperature{0.8f};
    float top_p{0.95f};
    int max_tokens{512};
    bool stream{false};
    std::optional<int> seed;
    // kv3d extension: reuse an existing paused session
    std::optional<std::string> session_id;
};

struct TokenUsage {
    int prompt_tokens{0};
    int completion_tokens{0};
    int total_tokens{0};
};

struct Choice {
    int index{0};
    ChatMessage message;
    std::string finish_reason;  // "stop" | "length" | "error"
};

struct ChatCompletionResponse {
    std::string id;
    std::string object{"chat.completion"};
    int64_t created{0};
    std::string model;
    std::vector<Choice> choices;
    TokenUsage usage;
    // kv3d extension: session ID for pause/resume
    std::optional<std::string> kv3d_session_id;
};

// ── Streaming delta (server-sent events) ─────────────────────────────────────

struct DeltaContent {
    std::string role;     // present only in first chunk
    std::string content;  // token(s) emitted this step
};

struct StreamChoice {
    int index{0};
    DeltaContent delta;
    std::optional<std::string> finish_reason;
};

struct ChatCompletionChunk {
    std::string id;
    std::string object{"chat.completion.chunk"};
    int64_t created{0};
    std::string model;
    std::vector<StreamChoice> choices;
};

// ── Error response ────────────────────────────────────────────────────────────

struct ErrorResponse {
    struct Error {
        std::string message;
        std::string type;
        std::optional<std::string> param;
        std::optional<std::string> code;
    } error;
};

}  // namespace kv3d::api
