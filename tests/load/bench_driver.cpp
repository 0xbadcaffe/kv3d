/// KV3D benchmark driver.
///
/// Measures:
///   - Prefix cache hit rate under shared-system-prompt workload
///   - Throughput (requests/sec)
///   - Memory per session (approximate, via stats)
///   - Exports results as CSV to stdout or a file

#include "kv3d/api/types.hpp"
#include "kv3d/metrics/metrics.hpp"
#include "kv3d/sched/session_manager.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace kv3d;
using namespace kv3d::api;

struct BenchConfig {
    int n_requests{1000};
    int max_tokens{64};
    float shared_prefix_ratio{0.8f};  // fraction of requests with the shared system prompt
    std::optional<std::string> csv_output;
};

static BenchConfig parse_args(int argc, char* argv[]) {
    BenchConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (arg == "--requests") cfg.n_requests = std::atoi(next());
        else if (arg == "--max-tokens") cfg.max_tokens = std::atoi(next());
        else if (arg == "--shared-ratio") cfg.shared_prefix_ratio = std::atof(next());
        else if (arg == "--output") cfg.csv_output = next();
    }
    return cfg;
}

int main(int argc, char* argv[]) {
    const BenchConfig cfg = parse_args(argc, argv);

    std::cout << "kv3d benchmark\n";
    std::cout << "  requests     : " << cfg.n_requests << "\n";
    std::cout << "  max_tokens   : " << cfg.max_tokens << "\n";
    std::cout << "  shared_ratio : " << cfg.shared_prefix_ratio << "\n\n";

    Metrics::instance().reset();

    SessionManagerConfig sm_cfg;
    SessionManager mgr(sm_cfg);

    const std::string shared_system = "You are a helpful AI assistant for enterprise use. "
                                      "Always be concise, factual, and professional.";
    const std::string alt_system = "You are a creative writing assistant.";

    const std::vector<std::string> user_messages = {
        "What is the capital of France?",
        "Explain quantum entanglement briefly.",
        "Write a haiku about clouds.",
        "Summarize the water cycle.",
        "What is 42 times 17?",
    };

    const auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < cfg.n_requests; ++i) {
        const bool use_shared =
            (static_cast<float>(i) / cfg.n_requests) < cfg.shared_prefix_ratio;
        const std::string& sys = use_shared ? shared_system : alt_system;
        const std::string& user = user_messages[i % user_messages.size()];

        ChatCompletionRequest req;
        req.model = "stub-model";
        req.max_tokens = cfg.max_tokens;
        req.messages = {{"system", sys}, {"user", user}};
        mgr.complete(req);
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_s =
        std::chrono::duration<double>(t1 - t0).count();
    const double rps = cfg.n_requests / elapsed_s;

    auto& metrics = Metrics::instance();
    const double hit_rate = metrics.prefix_hit_rate() * 100.0;

    std::cout << "Results:\n";
    std::cout << "  elapsed_s    : " << elapsed_s << "\n";
    std::cout << "  requests/sec : " << rps << "\n";
    std::cout << "  prefix hits  : " << metrics.prefix_hits() << "\n";
    std::cout << "  prefix misses: " << metrics.prefix_misses() << "\n";
    std::cout << "  hit_rate     : " << hit_rate << "%\n";
    std::cout << "  ram_cache_b  : " << metrics.ram_cache_bytes() << "\n";

    const std::string csv = metrics.csv_row();
    if (cfg.csv_output.has_value()) {
        std::ofstream f(*cfg.csv_output);
        f << csv;
        std::cout << "\nResults written to: " << *cfg.csv_output << "\n";
    } else {
        std::cout << "\nCSV:\n" << csv;
    }

    return 0;
}
