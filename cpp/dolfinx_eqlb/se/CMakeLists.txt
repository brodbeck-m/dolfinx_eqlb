set(HEADERS_se
    ${CMAKE_CURRENT_SOURCE_DIR}/KernelData.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Patch.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/deqlb_se.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.hpp
    PARENT_SCOPE
)

target_sources(
  dolfinx_eqlb
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/KernelData.cpp
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Patch.cpp
)