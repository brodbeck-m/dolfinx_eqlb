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
# configurable options for how we want to build

include(FeatureSummary)

option(BUILD_SHARED_LIBS "Build DOLFINX_EQLB with shared libraries." ON)
add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build DOLFINX_EQLB with shared libraries.")

option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath." ON)
add_feature_info(CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

# Find packages
find_package(DOLFINX REQUIRED)
find_package(Basix REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

feature_summary(WHAT ALL)

add_library(dolfinx_eqlb "")

target_link_libraries(dolfinx_eqlb PUBLIC Basix::basix)
target_link_libraries(dolfinx_eqlb PUBLIC dolfinx)

# Installation of DOLFINX_EQLB Library
include(GNUInstallDirs)

set(
  HEADERS_EQLB
  BoundaryData.hpp
  FluxBC.hpp
  KernelData.hpp
  Patch.hpp
  PatchFluxEV.hpp
  PatchFluxCstm.hpp
  ProblemData.hpp
  ProblemDataFluxEV.hpp
  ProblemDataFluxCstm.hpp
  QuadratureRule.hpp
  StorageStiffness.hpp
  assemble_patch_constrmin.hpp
  assemble_patch_semiexplt.hpp
  local_solver.hpp
  reconstruction.hpp
  solve_patch_constrmin.hpp
  solve_patch_semiexplt.hpp
  utils.hpp
)

target_sources(
  dolfinx_eqlb
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryData.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/KernelData.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Patch.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/PatchFluxEV.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
)

install(FILES ${HEADERS_EQLB} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx_eqlb COMPONENT Development)

# Set target include location (for build and installed)
target_include_directories(dolfinx_eqlb PUBLIC
                           $<INSTALL_INTERFACE:include>
                           "$<BUILD_INTERFACE:${DOLFINX_EQLB_SOURCE_DIR};${DOLFINX_EQLB_SOURCE_DIR}/dolfinx_eqlb>")

# Set target properties
set_target_properties(dolfinx_eqlb PROPERTIES
  VERSION ${DOLFINX_EQLB_VERSION}
  SOVERSION ${DOLFINX_EQLB_VERSION_MAJOR}.${DOLFINX_EQLB_VERSION_MINOR})

#------------------------------------------------------------------------------
# Install dolfinx library and header files
install(TARGETS dolfinx_eqlb
  EXPORT DOLFINX_EQLBTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
  )

# Generate DOLFINTargets.cmake
install(EXPORT DOLFINX_EQLBTargets DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_eqlb)

include(CMakePackageConfigHelpers)

configure_package_config_file(${DOLFINX_EQLB_SOURCE_DIR}/DOLFINX_EQLBConfig.cmake.in
${CMAKE_BINARY_DIR}/dolfinx_eqlb/DOLFINX_EQLBConfig.cmake
INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_eqlb)

# Install CMake helper files
install(
  FILES
  ${CMAKE_BINARY_DIR}/dolfinx_eqlb/DOLFINX_EQLBConfig.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_eqlb
  COMPONENT Development)