cmake_minimum_required(VERSION 3.8)
project("lib$ENV{PKG_NAME}" VERSION $ENV{PKG_VERSION} LANGUAGES CXX)

# Setup global options.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/info)
set(CMAKE_BUILD_SCRIPTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/info")

set(PKG_NAME $ENV{PKG_NAME})
set(CMAKE_PREFIX_PATH $ENV{PREFIX})
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/info)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

string(REPLACE "." "" FORMATTED_PY_VER "$ENV{PY_VER}") 
set(PKG_VERSION "$ENV{GIT_DESCRIBE_TAG}-py${FORMATTED_PY_VER}_$ENV{GIT_DESCRIBE_NUMBER}")

find_package(pybind11 REQUIRED)
find_package(Armadillo REQUIRED)
find_package(NumPy REQUIRED) 

set(SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/volpriceinference.cpp")

pybind11_add_module(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${NUMPY_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include
    $ENV{PREFIX}/include) 

# I create a file that tracks the current version.

set(MY_VERSION_FILENAME "${PKG_NAME}/version.py")
set(MY_VERSION_STRING "__version__ = \"${PKG_VERSION}\"")
file(GENERATE  OUTPUT "$<TARGET_FILE_DIR:${PROJECT_NAME}>/${MY_VERSION_FILENAME}" CONTENT "${MY_VERSION_STRING}")
add_custom_target(VERSION_FILE ALL DEPENDS "${MY_VERSION_FILANEM}"
    COMMENT "I am writing the version ${MY_VERSION_STRING} to the version file ${MY_VERSION_FILENAME}.")

target_compile_features(${PROJECT_NAME}  PRIVATE cxx_std_17)
target_compile_options(${PROJECT_NAME} PUBLIC -fdiagnostics-color=always)

message(STATUS "The build type is ${CMAKE_BUILD_TYPE}")                                                        
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")                                                                      
    set(MY_RELEASE_COMPILE_OPTIONS -mtune=native -mfpmath=sse -funroll-loops, -ffast-math)
    target_compile_options(${PROJECT_NAME} PUBLIC ${MY_RELEASE_COMPILE_OPTIONS})                                      
elseif ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")                                                                         
    set(MY_DEBUG_COMPILE_OPTIONS -Wpedantic -Werror -Wall -Wextra)
    target_compile_options(${PROJECT_NAME} PRIVATE ${MY_DEBUG_COMPILE_OPTIONS})                                      
endif()      

target_link_libraries(${PROJECT_NAME} PUBLIC ${ARMADILLO_LIBRARIES})

# Call Clang-Tidy
find_program(CLANG_TIDY_EXE NAMES "clang-tidy")

if(NOT CLANG_TIDY_EXE)
  message(STATUS "clang-tidy not found.")
else()
  message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
  set(DO_CLANG_TIDY "${CLANG_TIDY_EXE}" "-checks=boost.*, bugprone.*, cppcoregidelines.*, llvm.*, modernize.*,  \
misc.*, clang-analyzer.*, mpi.*, performance.*, portability.*, readability.*, -clang-analyzer-alpha.*")

endif()

if(CLANG_TIDY_EXE)
  set_target_properties(
    ${PROJECT_NAME} PROPERTIES
    CXX_CLANG_TIDY "${DO_CLANG_TIDY}"
  )
endif()

