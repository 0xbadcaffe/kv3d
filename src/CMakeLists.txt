# ── kv3d_core ──────────────────────────────────────────────────────────────────
add_library(kv3d_core STATIC
    core/prefix_hash.cpp
    core/canonical_prompt.cpp
    core/guardrails.cpp
    core/fallback_policy.cpp
)
target_include_directories(kv3d_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(kv3d_core PUBLIC nlohmann_json::nlohmann_json)
kv3d_set_warnings(kv3d_core)

# ── kv3d_kv ────────────────────────────────────────────────────────────────────
add_library(kv3d_kv STATIC
    kv/prefix_store.cpp
    kv/delta_codec.cpp
    kv/quant_residual.cpp
)
target_include_directories(kv3d_kv PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(kv3d_kv PUBLIC kv3d_core)
kv3d_set_warnings(kv3d_kv)

# ── kv3d_storage ───────────────────────────────────────────────────────────────
add_library(kv3d_storage STATIC
    storage/gpu_cache.cpp
    storage/ram_cache.cpp
    storage/snapshot_index.cpp
)
target_include_directories(kv3d_storage PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(kv3d_storage PUBLIC kv3d_kv)
kv3d_set_warnings(kv3d_storage)

# ── kv3d_metrics ───────────────────────────────────────────────────────────────
add_library(kv3d_metrics STATIC
    metrics/metrics.cpp
    metrics/prom_exporter.cpp
)
target_include_directories(kv3d_metrics PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(kv3d_metrics PUBLIC kv3d_core httplib::httplib)
kv3d_set_warnings(kv3d_metrics)

# ── kv3d_sched ─────────────────────────────────────────────────────────────────
add_library(kv3d_sched STATIC
    sched/session_manager.cpp
)
target_include_directories(kv3d_sched PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(kv3d_sched PUBLIC kv3d_kv kv3d_storage kv3d_metrics)
kv3d_set_warnings(kv3d_sched)

# ── kv3d_api ───────────────────────────────────────────────────────────────────
add_library(kv3d_api STATIC
    api/server.cpp
    api/routes.cpp
    api/stream_writer.cpp
)
target_include_directories(kv3d_api PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(kv3d_api PUBLIC kv3d_sched httplib::httplib nlohmann_json::nlohmann_json)
kv3d_set_warnings(kv3d_api)

# ── kv3d_cli (binary) ──────────────────────────────────────────────────────────
add_executable(kv3d_cli
    cli/main.cpp
    cli/config.cpp
)
set_target_properties(kv3d_cli PROPERTIES OUTPUT_NAME kv3d)
target_link_libraries(kv3d_cli PRIVATE kv3d_api nlohmann_json::nlohmann_json)
kv3d_set_warnings(kv3d_cli)

if(KV3D_LLAMA_AVAILABLE)
    target_link_libraries(kv3d_sched PUBLIC llama)
    target_compile_definitions(kv3d_sched PUBLIC KV3D_LLAMA_AVAILABLE=1)
    target_compile_definitions(kv3d_cli PUBLIC KV3D_LLAMA_AVAILABLE=1)
endif()

if(KV3D_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(kv3d_storage PRIVATE CUDA::cudart)
    target_compile_definitions(kv3d_storage PRIVATE KV3D_CUDA=1)
endif()
