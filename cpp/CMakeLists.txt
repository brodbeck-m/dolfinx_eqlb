# Top level CMakeLists for dolfinx_eqlb
cmake_minimum_required(VERSION 3.19)

# Set project name and version number
project(DOLFINX_EQLB VERSION 1.2.0)

#------------------------------------------------------------------------------
# Use C++20
set(CMAKE_CXX_STANDARD 20)

# Require C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Do not enable compler-specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)

#------------------------------------------------------------------------------
# configurable options for how we want to build
include(FeatureSummary)

option(BUILD_SHARED_LIBS "Build DOLFINX_EQLB with shared libraries." ON)
add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build DOLFINX_EQLB with shared libraries.")

# Add shared library paths so shared libs in non-system paths are found
option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath." ON)
add_feature_info(CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}) # Not identical to dolfinx make

# Find packages
find_package(DOLFINX 0.6.0.0 REQUIRED)
find_package(Basix 0.6.0.0 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
feature_summary(WHAT ALL)

#------------------------------------------------------------------------------
# Installation of the dolfinx_eqlb library
add_subdirectory(dolfinx_eqlb)
