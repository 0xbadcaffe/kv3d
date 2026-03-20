#include "kv3d/core/guardrails.hpp"

// Fallback policy helpers — configuration and serialization utilities.
// The policy itself lives in Guardrails; this file holds factory helpers.

namespace kv3d {

/// Build a permissive policy suitable for development / testing.
FallbackPolicy make_lenient_policy() {
    return FallbackPolicy{
        .max_logit_drift = 0.5f,
        .max_perplexity_delta = 2.0f,
        .auto_fallback_enabled = false,
    };
}

/// Build a strict production policy.
FallbackPolicy make_strict_policy() {
    return FallbackPolicy{
        .max_logit_drift = 0.05f,
        .max_perplexity_delta = 0.2f,
        .auto_fallback_enabled = true,
    };
}

}  // namespace kv3d
