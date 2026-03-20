#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>

namespace kv3d {

/// Central metrics collector. All counters are thread-safe (atomic).
/// Expose via Prometheus exporter or CSV dump.
class Metrics {
public:
    static Metrics& instance() noexcept;

    // ── Prefix cache ──────────────────────────────────────────────────────────
    void record_prefix_hit() noexcept { prefix_hits_.fetch_add(1, std::memory_order_relaxed); }
    void record_prefix_miss() noexcept { prefix_misses_.fetch_add(1, std::memory_order_relaxed); }

    [[nodiscard]] uint64_t prefix_hits() const noexcept {
        return prefix_hits_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] uint64_t prefix_misses() const noexcept {
        return prefix_misses_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] double prefix_hit_rate() const noexcept;

    // ── Sessions ──────────────────────────────────────────────────────────────
    void record_session_created() noexcept {
        sessions_created_.fetch_add(1, std::memory_order_relaxed);
    }
    void record_session_resumed() noexcept {
        sessions_resumed_.fetch_add(1, std::memory_order_relaxed);
    }
    void record_fallback() noexcept { fallbacks_.fetch_add(1, std::memory_order_relaxed); }

    [[nodiscard]] uint64_t sessions_created() const noexcept {
        return sessions_created_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] uint64_t fallbacks() const noexcept {
        return fallbacks_.load(std::memory_order_relaxed);
    }

    // ── Memory ───────────────────────────────────────────────────────────────
    void set_gpu_cache_bytes(size_t bytes) noexcept;
    void set_ram_cache_bytes(size_t bytes) noexcept;
    [[nodiscard]] size_t gpu_cache_bytes() const noexcept;
    [[nodiscard]] size_t ram_cache_bytes() const noexcept;

    // ── Latency helpers ───────────────────────────────────────────────────────
    void record_prefill_latency_us(uint64_t us) noexcept;
    void record_resume_latency_us(uint64_t us) noexcept;
    [[nodiscard]] uint64_t p50_prefill_latency_us() const noexcept;
    [[nodiscard]] uint64_t p95_resume_latency_us() const noexcept;

    /// Dump all metrics as a Prometheus-style text exposition.
    [[nodiscard]] std::string prometheus_text() const;

    /// Dump all metrics as a single-line CSV header + row pair.
    [[nodiscard]] std::string csv_row() const;

    void reset() noexcept;

private:
    Metrics() = default;

    std::atomic<uint64_t> prefix_hits_{0};
    std::atomic<uint64_t> prefix_misses_{0};
    std::atomic<uint64_t> sessions_created_{0};
    std::atomic<uint64_t> sessions_resumed_{0};
    std::atomic<uint64_t> fallbacks_{0};
    std::atomic<size_t> gpu_cache_bytes_{0};
    std::atomic<size_t> ram_cache_bytes_{0};

    // Simple running sum for latency — p-values approximated from running histogram
    std::atomic<uint64_t> prefill_latency_sum_us_{0};
    std::atomic<uint64_t> prefill_latency_count_{0};
    std::atomic<uint64_t> resume_latency_sum_us_{0};
    std::atomic<uint64_t> resume_latency_count_{0};
};

}  // namespace kv3d
