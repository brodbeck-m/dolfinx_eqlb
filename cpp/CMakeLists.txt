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

# Declare the library
add_library(dolfinx_eqlb "")

target_link_libraries(dolfinx_eqlb PUBLIC Basix::basix)
target_link_libraries(dolfinx_eqlb PUBLIC dolfinx)

#------------------------------------------------------------------------------
# Installation of DOLFINX_EQLB Library
include(GNUInstallDirs)

#------------------------------------------------------------------------------
# Add source files
set(EQLB_DIRS
    base
)

set(
  HEADERS_EQLB
  BoundaryData.hpp
  FluxBC.hpp
  KernelData.hpp
  PatchFluxEV.hpp
  PatchCstm.hpp
  PatchData.hpp
  ProblemData.hpp
  ProblemDataFluxCstm.hpp
  ProblemDataFluxEV.hpp
  QuadratureRule.hpp
  StorageStiffness.hpp
  assemble_patch_constrmin.hpp
  assemble_patch_semiexplt.hpp
  assemble_projection_boundary.hpp
  local_solver.hpp
  reconstruction.hpp
  solve_patch_constrmin.hpp
  solve_patch_semiexplt.hpp
  solve_patch_weaksym.hpp
  utils.hpp
)

target_sources(
  dolfinx_eqlb
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryData.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/KernelData.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/PatchCstm.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/PatchFluxEV.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
)

foreach(DIR ${EQLB_DIRS})
  add_subdirectory(${DIR})
endforeach()

# Set target include location (for build and installed)
target_include_directories(dolfinx_eqlb PUBLIC
                           $<INSTALL_INTERFACE:include>
                           "$<BUILD_INTERFACE:${DOLFINX_EQLB_SOURCE_DIR};${DOLFINX_EQLB_SOURCE_DIR}/dolfinx_eqlb>")

# Set target properties
set_target_properties(
  dolfinx_eqlb PROPERTIES
  VERSION ${DOLFINX_EQLB_VERSION}
  SOVERSION ${DOLFINX_EQLB_VERSION_MAJOR}.${DOLFINX_EQLB_VERSION_MINOR}
  )

#------------------------------------------------------------------------------
# Install library and header files
install(
  TARGETS dolfinx_eqlb
  EXPORT DOLFINX_EQLBTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
  )

foreach(DIR ${EQLB_DIRS})
  message(STATUS "folder=${${DIR}}")
  install(
    FILES ${HEADERS_${DIR}}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx_eqlb/${DIR}
    COMPONENT Development
  )
endforeach()

install(
  FILES dolfinx_eqlb.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx_eqlb
  COMPONENT Development
)

install(
  FILES ${HEADERS_EQLB} 
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx_eqlb 
  COMPONENT Development
)

# Generate DOLFINX_EQLBTargets.cmake
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
  COMPONENT Development
)
