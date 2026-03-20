#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace kv3d {

/// Quality signal collected after a decode step.
struct QualitySignal {
    float logit_drift{0.0f};    // KL divergence vs raw-KV baseline (approximated)
    float perplexity_delta{0.0f};
    uint32_t token_index{0};
    uint64_t session_id{0};
};

/// Policy controlling when the engine falls back to raw KV.
struct FallbackPolicy {
    float max_logit_drift{0.15f};    // trigger if KL > this threshold
    float max_perplexity_delta{0.5f};
    bool  auto_fallback_enabled{true};
    /// If set, called on every guardrail decision for monitoring.
    std::function<void(uint64_t session_id, bool triggered, float drift)> on_decision;
};

enum class GuardrailVerdict { Pass, Fallback };

class Guardrails {
public:
    explicit Guardrails(FallbackPolicy policy = {});

    /// Evaluate a quality signal and return Pass or Fallback.
    [[nodiscard]] GuardrailVerdict evaluate(const QualitySignal& sig) const;

    /// Update policy at runtime (e.g., from config reload).
    void set_policy(FallbackPolicy policy);
    [[nodiscard]] const FallbackPolicy& policy() const noexcept;

private:
    FallbackPolicy policy_;
};

}  // namespace kv3d
