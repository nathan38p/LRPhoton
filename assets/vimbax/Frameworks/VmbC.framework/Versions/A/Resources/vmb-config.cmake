

if(${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
    set(_FIND_COMPONENTS_QUIET QUIET)
    macro(_vmb_find_message MESSAGE_TYPE)
        if (MESSAGE_TYPE STREQUAL "FATAL_ERROR")
            return()
        endif()
    endmacro()
else()
    macro(_vmb_find_message)
        message(${ARGN})
    endmacro()
endif()

if(NOT APPLE)
    _vmb_find_message(FATAL_ERROR "This find script is made for macOS only and will not work on ${CMAKE_SYSTEM_NAME}")
endif()

get_filename_component(FRAMEWORK_DIR "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)

# prerequesites of each component (VMB_<component>_DEPENDENCIES = include file name)
set(VMB_c_DEPENDENCIES)
set(VMB_cpp_DEPENDENCIES vmb_c)
set(VMB_imagetransform_DEPENDENCIES)

if(CMAKE_VERSION VERSION_LESS 3.0)
    _vmb_find_message(FATAL_ERROR "CMake versions < 3.0 are not supported")
endif()

if(NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS)
    set(${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS "C") # fallback default case
endif()

set(_ALL_FOUND TRUE)
foreach(_COMP IN LISTS ${CMAKE_FIND_PACKAGE_NAME}_FIND_COMPONENTS)
    if(EXISTS "${FRAMEWORK_DIR}/Vmb${_COMP}.framework/" AND (${_COMP} STREQUAL "C" OR ${_COMP} STREQUAL "CPP" OR ${_COMP} STREQUAL "ImageTransform"))
        if (NOT TARGET Vmb::${_COMP})
            add_library(Vmb::${_COMP} SHARED IMPORTED)
            set_target_properties(Vmb::${_COMP} PROPERTIES 
                IMPORTED_LOCATION "${FRAMEWORK_DIR}/Vmb${_COMP}.framework/Vmb${_COMP}" # CMake doc: For frameworks on macOS this is the location of the library file symlink just inside the framework folder.
                FRAMEWORK TRUE
                FRAMEWORK_DIR "${FRAMEWORK_DIR}/Vmb${_COMP}.framework"
                INTERFACE_INCLUDE_DIRECTORIES "${FRAMEWORK_DIR}/Vmb${_COMP}.framework"
            )
            set(VMB_COMPONENT_${_COMP}_FOUND True)
        endif()
    else()
        set(_ALL_FOUND FALSE)
        set(VMB_COMPONENT_${_COMP}_FOUND FALSE)
        if (${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED_${_COMP})
            _vmb_find_message(FATAL_ERROR "Unknown Vmb component required: ${_COMP}")
        else()
            _vmb_find_message(WARNING "Unknown Vmb component: ${_COMP}")
        endif()
    endif()
endforeach()
set(${CMAKE_FIND_PACKAGE_NAME}_FOUND ${_ALL_FOUND})
