#pragma once

#include "kv3d/api/server.hpp"
#include "kv3d/sched/session_manager.hpp"

#include <filesystem>
#include <optional>
#include <string>

namespace kv3d {

struct EngineConfig {
    api::ServerConfig server;
    SessionManagerConfig session;
    std::string model_path;
    std::string model_id{"default"};
};

EngineConfig load_config(std::optional<std::string> path_override = std::nullopt);
void write_default_config(const std::filesystem::path& path);

}  // namespace kv3d
