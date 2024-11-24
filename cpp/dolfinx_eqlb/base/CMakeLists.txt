set(HEADERS_base
    ${CMAKE_CURRENT_SOURCE_DIR}/KernelData.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Patch.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/QuadratureRule.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/deqlb_base.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mdspan.hpp
    PARENT_SCOPE
)

target_sources(
  dolfinx_eqlb
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/KernelData.cpp
)