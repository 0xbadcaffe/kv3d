#include "kv3d/kv/kv_block.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace kv3d {

/// Compute the per-tensor L2 norm of the delta (base vs full).
/// Used by the guardrails system to estimate quality drift.
float compute_delta_l2(std::span<const float> base, std::span<const float> full) noexcept {
    if (base.size() != full.size() || base.empty()) return 0.0f;

    float sum_sq = 0.0f;
    for (size_t i = 0; i < base.size(); ++i) {
        const float d = full[i] - base[i];
        sum_sq += d * d;
    }
    return std::sqrt(sum_sq / static_cast<float>(base.size()));
}

/// Approximate KL divergence between softmax(base_logits) and softmax(full_logits).
/// Used as a fast proxy for output quality drift.
float approximate_kl_divergence(std::span<const float> base_logits,
                                std::span<const float> full_logits) noexcept {
    if (base_logits.size() != full_logits.size() || base_logits.empty()) return 0.0f;

    // Find max for numerical stability
    float max_b = *std::max_element(base_logits.begin(), base_logits.end());
    float max_f = *std::max_element(full_logits.begin(), full_logits.end());

    // Softmax and KL
    float sum_b = 0.0f, sum_f = 0.0f;
    const size_t n = base_logits.size();
    std::vector<float> p(n), q(n);

    for (size_t i = 0; i < n; ++i) {
        p[i] = std::exp(base_logits[i] - max_b);
        q[i] = std::exp(full_logits[i] - max_f);
        sum_b += p[i];
        sum_f += q[i];
    }

    float kl = 0.0f;
    constexpr float eps = 1e-10f;
    for (size_t i = 0; i < n; ++i) {
        const float pi = p[i] / sum_b;
        const float qi = q[i] / sum_f;
        if (pi > eps) kl += pi * std::log(pi / (qi + eps));
    }
    return kl;
}

}  // namespace kv3d
