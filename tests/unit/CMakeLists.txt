add_executable(kv3d_unit_tests
    test_prefix_hash.cpp
    test_delta_codec.cpp
)
target_link_libraries(kv3d_unit_tests
    PRIVATE
        kv3d_core
        kv3d_kv
        Catch2::Catch2WithMain
)
catch_discover_tests(kv3d_unit_tests)

# Phase 2 experimental tests — separate binary so the baseline suite stays clean
add_executable(kv3d_experimental_tests
    test_group_codec.cpp
)
target_link_libraries(kv3d_experimental_tests
    PRIVATE
        kv3d_kv_experimental
        Catch2::Catch2WithMain
)
catch_discover_tests(kv3d_experimental_tests)
