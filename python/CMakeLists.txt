cmake_minimum_required(VERSION 3.21)

set(PROJECT_NAME dolfinx_eqlb_nanobind)
project(${PROJECT_NAME})

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Detect Eigen3
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# Detect DOLFINx and Basix
execute_process(
  COMMAND
  ${Python3_EXECUTABLE} -c "import os, sys, basix; sys.stdout.write(os.path.dirname(basix.__file__))"
  OUTPUT_VARIABLE BASIX_PY_DIR
  RESULT_VARIABLE BASIX_PY_COMMAND_RESULT
  ERROR_VARIABLE BASIX_ERROR_OUT OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(Basix REQUIRED CONFIG HINTS ${BASIX_PY_DIR})
if (Basix_FOUND)
  message(STATUS "Found Basix at ${Basix_DIR}")
endif()

find_package(DOLFINX REQUIRED CONFIG)
if (DOLFINX_FOUND)
  message(STATUS "Found DOLFINx at ${DOLFINX_DIR}")
endif()

# Add dolfinx_eqlb libraries
message(STATUS "Module patch ${CMAKE_MODULE_PATH}")
find_package(DOLFINX_EQLB REQUIRED CONFIG)
if (DOLFINX_EQLB_FOUND)
  message(STATUS "Found dolfinx_eqlb at ${DOLFINX_EQLB_DIR}")
endif()

# Create the binding library nanobind handles its own calls to target_link_libraries
nanobind_add_module(
  cpp
  NOMINSIZE
  MODULE
  dolfinx_eqlb/wrappers.cpp
)

target_compile_definitions(cpp PRIVATE DOLFINX_EQLB_VERSION="${DOLFINX_EQLB_VERSION}")
target_link_libraries(cpp PUBLIC Eigen3::Eigen)
target_link_libraries(cpp PUBLIC dolfinx)
target_link_libraries(cpp PUBLIC basix)
target_link_libraries(cpp PRIVATE dolfinx_eqlb)

# Check for petsc4py
execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import petsc4py; print(petsc4py.get_include())"
  OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
  RESULT_VARIABLE PETSC4PY_COMMAND_RESULT
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PETSC4PY_COMMAND_RESULT)
  message(STATUS "Found petsc4py include directory at ${PETSC4PY_INCLUDE_DIR}")
  target_include_directories(cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})
  target_compile_definitions(cpp PRIVATE HAS_PETSC4PY)
else()
  message(FATAL_ERROR "petsc4py not found.")
endif()

# Get python include-dirs
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import dolfinx.wrappers, sys; sys.stdout.write(str(dolfinx.wrappers.get_include_path()))"
  OUTPUT_VARIABLE DOLFINX_PY_DIR
  RESULT_VARIABLE DOLFINX_PY_COMMAND_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)

if (DOLFINX_PY_DIR)
  message(STATUS "Adding ${DOLFINX_PY_DIR} to include directories")
  target_include_directories(cpp PRIVATE ${DOLFINX_PY_DIR})
endif()

set_target_properties(cpp PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS cpp DESTINATION dolfinx_eqlb)