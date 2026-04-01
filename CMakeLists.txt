cmake_minimum_required(VERSION 3.14)

project(pong)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(SFML_DIR "libs/SFML")
set(EIGEN_DIR "libs/eigen")

add_subdirectory(libs/SFML)

include_directories(${CMAKE_SOURCE_DIR}
    include
    ${EIGEN_DIR}
)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/pseument.cpp
    src/convolayer.cpp
    src/denselayer.cpp
)


target_link_libraries(${PROJECT_NAME}
    PRIVATE
    sfml-system
    sfml-window
    sfml-graphics
    sfml-audio
    sfml-network
)

