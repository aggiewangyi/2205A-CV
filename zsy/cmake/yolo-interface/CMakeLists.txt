cmake_minimum_required(VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(PROJECT_NAME "trt_inference")
project(${PROJECT_NAME})
add_definitions(-std=c++11)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")
 
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)
 
set(SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
add_executable(${PROJECT_NAME}  ${SOURCE_DIR}/cpp_tensorrt_yolo.cpp  ${SOURCE_DIR}/result.cpp  ${SOURCE_DIR}/result.h  ${SOURCE_DIR}/dirent.h)   #4

# setup CUDA
find_package(CUDA REQUIRED)
message(STATUS "   cuda libraries: ${CUDA_LIBRARIES}")
message(STATUS "   cuda include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
enable_language(CUDA)  # add this line, then no need to setup cuda path in vs


set(OpenCV_DIR "E:/PycharmProject/OpenCV/opencv-4.6.0-vc14_vc15/opencv/build/x64/vc15/lib")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})


set(TRT_DIR "C:/Program Files/TensorRT-8.6.1.6")  #4
include_directories(${TRT_DIR}/include)
set(TRT_LINK_DIR ${TRT_DIR}/lib)
link_directories(${TRT_LINK_DIR}) #7


target_link_libraries(${PROJECT_NAME} "nvinfer" "nvinfer_plugin" "nvonnxparser")   #9
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})          #10
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})       #11


message(STATUS "well done!")