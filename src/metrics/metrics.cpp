#include "kv3d/metrics/metrics.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace kv3d {

Metrics& Metrics::instance() noexcept {
    static Metrics inst;
    return inst;
}

double Metrics::prefix_hit_rate() const noexcept {
    const uint64_t hits = prefix_hits_.load(std::memory_order_relaxed);
    const uint64_t misses = prefix_misses_.load(std::memory_order_relaxed);
    const uint64_t total = hits + misses;
    return total == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(total);
}

void Metrics::set_gpu_cache_bytes(size_t bytes) noexcept {
    gpu_cache_bytes_.store(bytes, std::memory_order_relaxed);
}
void Metrics::set_ram_cache_bytes(size_t bytes) noexcept {
    ram_cache_bytes_.store(bytes, std::memory_order_relaxed);
}
size_t Metrics::gpu_cache_bytes() const noexcept {
    return gpu_cache_bytes_.load(std::memory_order_relaxed);
}
size_t Metrics::ram_cache_bytes() const noexcept {
    return ram_cache_bytes_.load(std::memory_order_relaxed);
}

void Metrics::record_prefill_latency_us(uint64_t us) noexcept {
    prefill_latency_sum_us_.fetch_add(us, std::memory_order_relaxed);
    prefill_latency_count_.fetch_add(1, std::memory_order_relaxed);
}
void Metrics::record_resume_latency_us(uint64_t us) noexcept {
    resume_latency_sum_us_.fetch_add(us, std::memory_order_relaxed);
    resume_latency_count_.fetch_add(1, std::memory_order_relaxed);
}

// Approximate p50 as the mean (good enough for MVP trending).
uint64_t Metrics::p50_prefill_latency_us() const noexcept {
    const uint64_t cnt = prefill_latency_count_.load(std::memory_order_relaxed);
    if (cnt == 0) return 0;
    return prefill_latency_sum_us_.load(std::memory_order_relaxed) / cnt;
}
uint64_t Metrics::p95_resume_latency_us() const noexcept {
    const uint64_t cnt = resume_latency_count_.load(std::memory_order_relaxed);
    if (cnt == 0) return 0;
    // Approximate p95 as mean * 2.0 (conservative for MVP; replace with HDR histogram later).
    return (resume_latency_sum_us_.load(std::memory_order_relaxed) / cnt) * 2;
}

std::string Metrics::prometheus_text() const {
    std::ostringstream o;
    auto g = [&](const char* name, const char* help, auto val) {
        o << "# HELP " << name << " " << help << "\n";
        o << "# TYPE " << name << " gauge\n";
        o << name << " " << val << "\n";
    };
    auto c = [&](const char* name, const char* help, auto val) {
        o << "# HELP " << name << " " << help << "\n";
        o << "# TYPE " << name << " counter\n";
        o << name << "_total " << val << "\n";
    };

    c("kv3d_prefix_hits", "Number of prefix cache hits", prefix_hits());
    c("kv3d_prefix_misses", "Number of prefix cache misses", prefix_misses());
    g("kv3d_prefix_hit_rate", "Fraction of requests that hit the prefix cache",
      prefix_hit_rate());
    c("kv3d_sessions_created", "Total sessions created", sessions_created());
    c("kv3d_fallback_events", "Total guardrail fallback events", fallbacks());
    g("kv3d_gpu_cache_bytes", "Bytes used in GPU hot cache", gpu_cache_bytes());
    g("kv3d_ram_cache_bytes", "Bytes used in RAM warm cache", ram_cache_bytes());
    g("kv3d_prefill_latency_p50_us", "Approximate p50 prefill latency (µs)",
      p50_prefill_latency_us());
    g("kv3d_resume_latency_p95_us", "Approximate p95 resume latency (µs)",
      p95_resume_latency_us());
    return o.str();
}

std::string Metrics::csv_row() const {
    std::ostringstream header, row;
    header << "prefix_hits,prefix_misses,hit_rate,sessions_created,fallbacks,"
              "gpu_cache_bytes,ram_cache_bytes,prefill_p50_us,resume_p95_us\n";
    row << prefix_hits() << "," << prefix_misses() << "," << prefix_hit_rate() << ","
        << sessions_created() << "," << fallbacks() << "," << gpu_cache_bytes() << ","
        << ram_cache_bytes() << "," << p50_prefill_latency_us() << ","
        << p95_resume_latency_us() << "\n";
    return header.str() + row.str();
}

void Metrics::reset() noexcept {
    prefix_hits_.store(0);
    prefix_misses_.store(0);
    sessions_created_.store(0);
    sessions_resumed_.store(0);
    fallbacks_.store(0);
    gpu_cache_bytes_.store(0);
    ram_cache_bytes_.store(0);
    prefill_latency_sum_us_.store(0);
    prefill_latency_count_.store(0);
    resume_latency_sum_us_.store(0);
    resume_latency_count_.store(0);
}

}  // namespace kv3d
