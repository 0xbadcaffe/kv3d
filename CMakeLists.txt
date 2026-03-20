cmake_minimum_required(VERSION 3.25)
project(kv3d-engine VERSION 0.1.0 LANGUAGES CXX)

# ── Language baseline ──────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Options ────────────────────────────────────────────────────────────────────
option(KV3D_BUILD_TESTS      "Build unit and integration tests"   ON)
option(KV3D_BUILD_BENCH      "Build benchmark runner"             ON)
option(KV3D_ENABLE_SANITIZERS "Enable ASan/UBSan (debug only)"   OFF)
option(KV3D_USE_CUDA         "Enable CUDA KV-cache hot tier"      OFF)
option(KV3D_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# ── CMake module path ──────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CompilerWarnings)
if(KV3D_ENABLE_SANITIZERS)
    include(Sanitizers)
endif()

# ── External dependencies (FetchContent) ──────────────────────────────────────
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# cpp-httplib — header-only HTTP/1.1 server/client
FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(httplib)

# nlohmann/json — header-only JSON library
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(json)

# Catch2 — unit-test framework (only when tests are enabled)
if(KV3D_BUILD_TESTS OR KV3D_BUILD_BENCH)
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.7.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(CTest)
    include(Catch)
endif()

# ── llama.cpp backend ──────────────────────────────────────────────────────────
if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
    set(KV3D_LLAMA_AVAILABLE TRUE)
    message(STATUS "kv3d: llama.cpp backend enabled (submodule)")
else()
    include(Findllama.cpp)
    if(LLAMA_FOUND)
        set(KV3D_LLAMA_AVAILABLE TRUE)
        message(STATUS "kv3d: llama.cpp backend enabled (system)")
    else()
        set(KV3D_LLAMA_AVAILABLE FALSE)
        message(WARNING "kv3d: llama.cpp not found — building with stub backend only")
    endif()
endif()

# ── Global include dirs ────────────────────────────────────────────────────────
include_directories(include)

# ── Subdirectories ─────────────────────────────────────────────────────────────
add_subdirectory(src)

if(KV3D_BUILD_TESTS OR KV3D_BUILD_BENCH)
    add_subdirectory(tests)
endif()

# ── Install rules ──────────────────────────────────────────────────────────────
install(TARGETS kv3d_cli RUNTIME DESTINATION bin)
install(DIRECTORY include/kv3d DESTINATION include)
