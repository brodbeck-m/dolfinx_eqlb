cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME dolfinx_eqlb_pybind11)
project(${PROJECT_NAME})

# Set C++ standard before finding pybind11
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED NO_MODULE) # Is this realy neccessary???

find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
  $ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})


find_package(DOLFINX REQUIRED)
find_package(DOLFINX_EQLB REQUIRED)


# Create the binding library
pybind11_add_module(cpp SHARED 
  dolfinx_eqlb/wrappers.cpp)


target_link_libraries(cpp PUBLIC pybind11::module)
target_link_libraries(cpp PUBLIC dolfinx)
target_link_libraries(cpp PUBLIC basix)
target_link_libraries(cpp PRIVATE dolfinx_eqlb)


# Get python include-dirs
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import dolfinx.wrappers, sys; sys.stdout.write(str(dolfinx.wrappers.get_include_path()))"
  OUTPUT_VARIABLE DOLFINX_PY_DIR
  RESULT_VARIABLE DOLFINX_PY_COMMAND_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)

if (DOLFINX_PY_DIR)
  message(STATUS "Adding ${DOLFINX_PY_DIR} to include directories")
  target_include_directories(cpp PRIVATE ${DOLFINX_PY_DIR})
endif()

   
# Find petsc4py through python
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import petsc4py; print(petsc4py.get_include())"
  OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
  RESULT_VARIABLE PETSC4PY_NOT_FOUND
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
target_include_directories(cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})


# Add to CMake search path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})