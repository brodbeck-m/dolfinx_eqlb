include(GNUInstallDirs)

#------------------------------------------------------------------------------
# Declare the library
add_library(dolfinx_eqlb "")

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

# ------------------------------------------------------------------------------
# Add include directories and libraries of required packages
target_link_libraries(dolfinx_eqlb PUBLIC Basix::basix)
target_link_libraries(dolfinx_eqlb PUBLIC dolfinx)

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
