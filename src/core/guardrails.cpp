#include "kv3d/core/guardrails.hpp"

#include <cmath>

namespace kv3d {

Guardrails::Guardrails(FallbackPolicy policy) : policy_(std::move(policy)) {}

GuardrailVerdict Guardrails::evaluate(const QualitySignal& sig) const {
    if (!policy_.auto_fallback_enabled) {
        if (policy_.on_decision) policy_.on_decision(sig.session_id, false, sig.logit_drift);
        return GuardrailVerdict::Pass;
    }

    const bool triggered = (sig.logit_drift > policy_.max_logit_drift) ||
                           (sig.perplexity_delta > policy_.max_perplexity_delta);

    if (policy_.on_decision) policy_.on_decision(sig.session_id, triggered, sig.logit_drift);

    return triggered ? GuardrailVerdict::Fallback : GuardrailVerdict::Pass;
}

void Guardrails::set_policy(FallbackPolicy policy) { policy_ = std::move(policy); }

const FallbackPolicy& Guardrails::policy() const noexcept { return policy_; }

}  // namespace kv3d
