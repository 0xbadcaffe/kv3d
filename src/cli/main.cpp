#include "kv3d/cli/config.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

void print_version() { std::cout << "kv3d 0.1.0 (MVP)\n"; }

void print_usage() {
    std::cout << R"(Usage: kv3d <command> [options]

Commands:
  serve     Start the inference server
  doctor    Check system requirements and configuration
  version   Print version information

serve options:
  --model   <path>   Path to a GGUF model file (required)
  --host    <addr>   Listen address (default: 127.0.0.1)
  --port    <n>      Listen port    (default: 8080)
  --threads <n>      Worker threads (default: 4)
  --config  <path>   Config file path

doctor options:
  (none)

Examples:
  kv3d serve --model ./models/qwen2.5-7b-instruct.Q4_K_M.gguf
  kv3d doctor
)";
}

// ── Minimal arg parser ────────────────────────────────────────────────────────

struct Args {
    std::string command;
    std::string model_path;
    std::string host;
    int port{0};
    int threads{0};
    std::string config_path;
};

Args parse_args(std::span<char*> argv) {
    Args a;
    if (argv.size() < 2) return a;
    a.command = argv[1];

    for (size_t i = 2; i < argv.size(); ++i) {
        std::string_view arg{argv[i]};
        auto next = [&]() -> std::string_view {
            return (i + 1 < argv.size()) ? argv[++i] : "";
        };

        if (arg == "--model") a.model_path = std::string(next());
        else if (arg == "--host") a.host = std::string(next());
        else if (arg == "--port") a.port = std::atoi(next().data());
        else if (arg == "--threads") a.threads = std::atoi(next().data());
        else if (arg == "--config") a.config_path = std::string(next());
    }
    return a;
}

// ── doctor command ────────────────────────────────────────────────────────────

int cmd_doctor() {
    std::cout << "kv3d doctor\n";
    std::cout << "===========\n";

    // CPU check
    std::cout << "[ok] CPU: ";
#if defined(__x86_64__)
    std::cout << "x86_64\n";
#elif defined(__aarch64__)
    std::cout << "aarch64\n";
#else
    std::cout << "unknown\n";
#endif

    // llama.cpp availability
#ifdef KV3D_LLAMA_AVAILABLE
    std::cout << "[ok] llama.cpp backend: available\n";
#else
    std::cout << "[warn] llama.cpp backend: NOT found (stub backend active)\n";
    std::cout << "       Run: git submodule update --init third_party/llama.cpp\n";
#endif

    // CUDA check
#ifdef KV3D_USE_CUDA
    std::cout << "[ok] CUDA: enabled\n";
#else
    std::cout << "[info] CUDA: disabled (CPU-only mode)\n";
#endif

    // Config file
    const char* xdg = std::getenv("XDG_CONFIG_HOME");
    const char* home = std::getenv("HOME");
    std::filesystem::path cfg = xdg ? (std::filesystem::path(xdg) / "kv3d" / "config.json")
                                    : (std::filesystem::path(home) / ".config" / "kv3d" / "config.json");
    if (std::filesystem::exists(cfg)) {
        std::cout << "[ok] config: " << cfg << "\n";
    } else {
        std::cout << "[info] config: not found at " << cfg << "\n";
        std::cout << "       Creating default config...\n";
        kv3d::write_default_config(cfg);
        std::cout << "       Written: " << cfg << "\n";
    }

    std::cout << "\nAll checks complete.\n";
    return 0;
}

// ── serve command ─────────────────────────────────────────────────────────────

int cmd_serve(const Args& args) {
    kv3d::EngineConfig cfg = kv3d::load_config(args.config_path.empty()
                                                   ? std::nullopt
                                                   : std::optional<std::string>(args.config_path));

    // CLI overrides take priority over config file.
    if (!args.model_path.empty()) cfg.model_path = args.model_path;
    if (!args.host.empty()) cfg.server.host = args.host;
    if (args.port > 0) cfg.server.port = args.port;
    if (args.threads > 0) cfg.server.thread_count = args.threads;

    if (cfg.model_path.empty()) {
        std::cerr << "Error: --model <path> is required (or set model.path in config)\n";
        return 1;
    }
    if (!std::filesystem::exists(cfg.model_path)) {
        std::cerr << "Error: model file not found: " << cfg.model_path << "\n";
        return 1;
    }

    std::cout << "kv3d engine starting\n";
    std::cout << "  model:   " << cfg.model_path << "\n";
    std::cout << "  listen:  " << cfg.server.host << ":" << cfg.server.port << "\n";
    std::cout << "  threads: " << cfg.server.thread_count << "\n";

    auto session_mgr = std::make_shared<kv3d::SessionManager>(cfg.session);
    kv3d::api::Server server(cfg.server, session_mgr);

    std::cout << "Ready. Press Ctrl+C to stop.\n";
    server.run();
    return 0;
}

}  // anonymous namespace

int main(int argc, char* argv[]) {
    const Args args = parse_args(std::span<char*>(argv, static_cast<size_t>(argc)));

    if (args.command.empty() || args.command == "--help" || args.command == "-h") {
        print_usage();
        return 0;
    }
    if (args.command == "version" || args.command == "--version") {
        print_version();
        return 0;
    }
    if (args.command == "doctor") return cmd_doctor();
    if (args.command == "serve") return cmd_serve(args);

    std::cerr << "Unknown command: " << args.command << "\n";
    print_usage();
    return 1;
}
