set(HEADERS_ev
    ${CMAKE_CURRENT_SOURCE_DIR}/Patch.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ProblemData.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/StorageStiffness.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/assembly.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/deqlb_ev.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/reconstruction.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/solve_patch.hpp
    PARENT_SCOPE
)

target_sources(
  dolfinx_eqlb
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Patch.cpp
)