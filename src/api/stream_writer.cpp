#include "kv3d/api/types.hpp"

#include <nlohmann/json.hpp>
#include <string>

namespace kv3d::api {

using json = nlohmann::json;

// ── JSON serialisation helpers ────────────────────────────────────────────────

static json message_to_json(const ChatMessage& msg) {
    return json{{"role", msg.role}, {"content", msg.content}};
}

/// Serialize a streaming chunk to a Server-Sent Events data line.
/// Format:  data: <json>\n\n
std::string chunk_to_sse(const ChatCompletionChunk& chunk) {
    json j;
    j["id"] = chunk.id;
    j["object"] = chunk.object;
    j["created"] = chunk.created;
    j["model"] = chunk.model;

    json choices_arr = json::array();
    for (const auto& c : chunk.choices) {
        json choice;
        choice["index"] = c.index;
        json delta;
        if (!c.delta.role.empty()) delta["role"] = c.delta.role;
        delta["content"] = c.delta.content;
        choice["delta"] = delta;
        if (c.finish_reason.has_value()) {
            choice["finish_reason"] = *c.finish_reason;
        } else {
            choice["finish_reason"] = nullptr;
        }
        choices_arr.push_back(choice);
    }
    j["choices"] = choices_arr;

    return "data: " + j.dump() + "\n\n";
}

/// The terminal SSE event.
std::string sse_done_sentinel() { return "data: [DONE]\n\n"; }

/// Serialize a full (non-streaming) ChatCompletionResponse to JSON.
std::string response_to_json(const ChatCompletionResponse& resp) {
    json j;
    j["id"] = resp.id;
    j["object"] = resp.object;
    j["created"] = resp.created;
    j["model"] = resp.model;

    json choices_arr = json::array();
    for (const auto& c : resp.choices) {
        json choice;
        choice["index"] = c.index;
        choice["message"] = message_to_json(c.message);
        choice["finish_reason"] = c.finish_reason;
        choices_arr.push_back(choice);
    }
    j["choices"] = choices_arr;

    j["usage"] = {{"prompt_tokens", resp.usage.prompt_tokens},
                  {"completion_tokens", resp.usage.completion_tokens},
                  {"total_tokens", resp.usage.total_tokens}};

    if (resp.kv3d_session_id.has_value()) {
        j["kv3d_session_id"] = *resp.kv3d_session_id;
    }

    return j.dump();
}

/// Serialize an error to JSON.
std::string error_to_json(const ErrorResponse& err) {
    json j;
    j["error"]["message"] = err.error.message;
    j["error"]["type"] = err.error.type;
    if (err.error.param.has_value()) j["error"]["param"] = *err.error.param;
    if (err.error.code.has_value()) j["error"]["code"] = *err.error.code;
    return j.dump();
}

/// Parse a ChatCompletionRequest from a JSON body string.
/// Throws nlohmann::json::exception on parse failure.
ChatCompletionRequest request_from_json(const std::string& body) {
    const json j = json::parse(body);

    ChatCompletionRequest req;
    req.model = j.value("model", "");
    req.temperature = j.value("temperature", 0.8f);
    req.top_p = j.value("top_p", 0.95f);
    req.max_tokens = j.value("max_tokens", 512);
    req.stream = j.value("stream", false);

    if (j.contains("seed") && !j["seed"].is_null()) req.seed = j["seed"].get<int>();
    if (j.contains("session_id") && j["session_id"].is_string())
        req.session_id = j["session_id"].get<std::string>();

    for (const auto& m : j.at("messages")) {
        req.messages.push_back({m.value("role", ""), m.value("content", "")});
    }

    return req;
}

}  // namespace kv3d::api
