function(kv3d_enable_sanitizers target)
    if(MSVC)
        message(WARNING "Sanitizers not supported on MSVC")
        return()
    endif()

    set(SANITIZER_FLAGS
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
        -g
    )

    target_compile_options(${target} PRIVATE ${SANITIZER_FLAGS})
    target_link_options(${target} PRIVATE ${SANITIZER_FLAGS})

    message(STATUS "kv3d: ASan/UBSan enabled for target '${target}'")
endfunction()

if(KV3D_ENABLE_SANITIZERS)
    # Apply to all targets built after this point
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
endif()
