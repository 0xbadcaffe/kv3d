# Findllama.cpp.cmake
# Searches for a system-installed llama.cpp library.
#
# Defines:
#   LLAMA_FOUND        - TRUE if found
#   LLAMA_INCLUDE_DIRS - include directories
#   LLAMA_LIBRARIES    - libraries to link against
#   llama::llama       - imported target (if found)

find_path(LLAMA_INCLUDE_DIR
    NAMES llama.h
    HINTS
        /usr/local/include
        /usr/include
        $ENV{LLAMA_ROOT}/include
    PATH_SUFFIXES llama
)

find_library(LLAMA_LIBRARY
    NAMES llama
    HINTS
        /usr/local/lib
        /usr/lib
        $ENV{LLAMA_ROOT}/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(llama.cpp
    REQUIRED_VARS LLAMA_LIBRARY LLAMA_INCLUDE_DIR
)

if(llama.cpp_FOUND)
    set(LLAMA_FOUND TRUE)
    set(LLAMA_INCLUDE_DIRS ${LLAMA_INCLUDE_DIR})
    set(LLAMA_LIBRARIES ${LLAMA_LIBRARY})

    if(NOT TARGET llama::llama)
        add_library(llama::llama UNKNOWN IMPORTED)
        set_target_properties(llama::llama PROPERTIES
            IMPORTED_LOCATION "${LLAMA_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${LLAMA_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(LLAMA_INCLUDE_DIR LLAMA_LIBRARY)
