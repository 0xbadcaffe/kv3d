add_subdirectory(unit)
add_subdirectory(integration)

if(KV3D_BUILD_BENCH)
    add_subdirectory(load)
endif()
