#include "kv3d/api/server.hpp"
#include "kv3d/sched/session_manager.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

namespace kv3d {

using json = nlohmann::json;
namespace fs = std::filesystem;

struct EngineConfig {
    api::ServerConfig server;
    SessionManagerConfig session;
    std::string model_path;
    std::string model_id{"default"};
};

static fs::path default_config_path() {
    const char* xdg = std::getenv("XDG_CONFIG_HOME");
    fs::path base = xdg ? fs::path(xdg) : (fs::path(std::getenv("HOME")) / ".config");
    return base / "kv3d" / "config.json";
}

/// Load config from a JSON file. Returns defaults if the file does not exist.
EngineConfig load_config(std::optional<std::string> path_override = std::nullopt) {
    fs::path cfg_path = path_override ? fs::path(*path_override) : default_config_path();

    EngineConfig cfg;

    if (!fs::exists(cfg_path)) return cfg;

    std::ifstream f(cfg_path);
    if (!f) throw std::runtime_error("Cannot open config: " + cfg_path.string());

    const json j = json::parse(f);

    if (j.contains("server")) {
        const auto& s = j["server"];
        cfg.server.host = s.value("host", "127.0.0.1");
        cfg.server.port = s.value("port", 8080);
        cfg.server.thread_count = s.value("threads", 4);
        cfg.server.api_key = s.value("api_key", "");
    }

    if (j.contains("cache")) {
        const auto& c = j["cache"];
        if (c.contains("ram_warm_mb")) {
            cfg.session.max_prefix_store_bytes =
                static_cast<size_t>(c["ram_warm_mb"].get<int>()) * 1024 * 1024;
        }
    }

    if (j.contains("model")) {
        cfg.model_path = j["model"].value("path", "");
        cfg.model_id = j["model"].value("id", "default");
    }

    return cfg;
}

/// Write a default config file to disk.
void write_default_config(const fs::path& path) {
    fs::create_directories(path.parent_path());
    json j = {
        {"server", {{"host", "127.0.0.1"}, {"port", 8080}, {"threads", 4}, {"api_key", ""}}},
        {"cache",
         {{"gpu_hot_mb", 2048}, {"ram_warm_mb", 8192}, {"ssd_cold_path", ""}}},
        {"model", {{"path", ""}, {"id", "default"}}}};

    std::ofstream f(path);
    f << j.dump(4) << '\n';
}

}  // namespace kv3d
