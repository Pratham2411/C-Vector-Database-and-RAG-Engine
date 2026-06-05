cmake_minimum_required(VERSION 3.16)

project(MY_AI VERSION 1.0.0 LANGUAGES CXX)

add_executable(db
    src/main.cpp
    src/api_server.cpp
)

target_compile_features(db PRIVATE cxx_std_17)
target_include_directories(db PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(db PRIVATE /W4)
else()
    target_compile_options(db PRIVATE -Wall -Wextra -pedantic)
endif()

if(WIN32)
    target_link_libraries(db PRIVATE ws2_32)
endif()
