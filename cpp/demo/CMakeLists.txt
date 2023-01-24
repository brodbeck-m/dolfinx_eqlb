cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME test_eqlb)
set(CONTAINS_UFL 0)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native ${CMAKE_C_FLAGS} -Wall")

find_package(DOLFINX REQUIRED)
find_package(Basix REQUIRED)
find_package(DOLFINX_EQLB REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(CONTAINS_UFL EQUAL 1)
  set(ufl_file test_eqlb)

  # Add target to compile UFL files
  if (PETSC_SCALAR_COMPLEX EQUAL 1)
    set(SCALAR_TYPE "--scalar_type=double _Complex")
  endif()
  # Create compilation command
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ufl_file}.c
    COMMAND ffcx ${SCALAR_TYPE} ${CMAKE_CURRENT_SOURCE_DIR}/${ufl_file}.py -o ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ufl_file}.py
    COMMENT "Compiling ${ufl_file}.py"
  )

  add_executable(${PROJECT_NAME} main.cpp ${ufl_file}.c)
else()
  add_executable(${PROJECT_NAME} main.cpp)
endif() 

# Find Boost program_options
if(DEFINED ENV{BOOST_ROOT} OR DEFINED BOOST_ROOT)
  set(Boost_NO_SYSTEM_PATHS on)
endif()
set(Boost_USE_MULTITHREADED $ENV{BOOST_USE_MULTITHREADED})
set(Boost_VERBOSE TRUE)
find_package(Boost 1.70 REQUIRED program_options)


target_link_directories(${PROJECT_NAME} PRIVATE dolfinx_eqlb)
target_link_libraries(${PROJECT_NAME} PRIVATE dolfinx)

target_link_libraries(${PROJECT_NAME} PRIVATE Boost::program_options)