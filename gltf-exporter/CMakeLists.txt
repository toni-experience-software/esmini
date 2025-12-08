set(TARGET_GLTF_EXPORTER export_odr_to_gltf)

add_executable(${TARGET_GLTF_EXPORTER} export_odr_to_gltf.cpp)

if(MSVC)
    target_compile_definitions(${TARGET_GLTF_EXPORTER} PRIVATE -D_CRT_SECURE_NO_WARNINGS)
endif()

target_include_directories(${TARGET_GLTF_EXPORTER} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/libs
    ${ROAD_MANAGER_PATH}
    ${VIEWER_BASE_PATH}
    ${COMMON_MINI_PATH}
    ${EXTERNALS_OSG_INCLUDES}
    ${EXTERNALS_PUGIXML_PATH}
    ${EXTERNALS_FMT_INCLUDES}
)

target_link_libraries(${TARGET_GLTF_EXPORTER} PRIVATE
    RoadGeom
    RoadManager
    CommonMini
    ${OSG_LIBRARIES}
    ${FMT_LIBRARIES}
)

if(LINUX)
    target_link_libraries(${TARGET_GLTF_EXPORTER} PRIVATE -lstdc++fs)
endif()

# Installation rules (optional, mimicking other apps)
if(LINUX OR APPLE)
    install(TARGETS ${TARGET_GLTF_EXPORTER} DESTINATION bin)
endif()
