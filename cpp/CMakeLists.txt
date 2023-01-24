# Top level CMakeLists for dolfinx_eqlb
cmake_minimum_required(VERSION 3.10)

# Set project name and version number
project(DOLFINX_EQLB VERSION 0.0.0)

#------------------------------------------------------------------------------
# Set CMake options, see `cmake --help-policy CMP00xx`

cmake_policy(VERSION 3.10)
if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
if (POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

# Use C++17
set(CMAKE_CXX_STANDARD 20)

# Require C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Do not enable compler-specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)

#------------------------------------------------------------------------------
# General configuration
# Make sure CMake uses the correct DOLFINConfig.cmake for tests and demos
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${CMAKE_CURRENT_BINARY_DIR}/dolfinx_eqlb)

#------------------------------------------------------------------------------
# configurable options for how we want to build

include(FeatureSummary)

option(BUILD_SHARED_LIBS "Build DOLFINX_EQLB with shared libraries." ON)
add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build DOLFINX_EQLB with shared libraries.")

option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath." ON)
add_feature_info(CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath.")

# Find packages
find_package(DOLFINX REQUIRED)
find_package(Basix REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
#INCLUDE_DIRECTORIES('/usr/include/eigen3')

feature_summary(WHAT ALL)

add_library(dolfinx_eqlb INTERFACE)

target_link_libraries(dolfinx_eqlb INTERFACE Basix::basix)
target_include_directories(dolfinx_eqlb INTERFACE ${DOLFINX_EQLB_SOURCE_DIR})

# Installation of header-only DOLFINX_EQLB Library
include(GNUInstallDirs)
install(FILES local_solver.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx_eqlb COMPONENT Development)

# Create package config for CMAKE
include(CMakePackageConfigHelpers)

configure_package_config_file(${DOLFINX_EQLB_SOURCE_DIR}/DOLFINX_EQLBConfig.cmake.in
${CMAKE_BINARY_DIR}/dolfinx_eqlb/DOLFINX_EQLBConfig.cmake
INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_eqlb)

write_basic_package_version_file(${CMAKE_BINARY_DIR}/dolfinx_eqlb/DOLFINX_EQLBConfigVersion.cmake
VERSION 0.5.1.0 COMPATIBILITY SameMajorVersion)

# Install CMake helper files
install(
  FILES
  ${CMAKE_BINARY_DIR}/dolfinx_eqlb/DOLFINX_EQLBConfig.cmake
  ${CMAKE_BINARY_DIR}/dolfinx_eqlb/DOLFINX_EQLBConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_eqlb
  COMPONENT Development)

#------------------------------------------------------------------------------