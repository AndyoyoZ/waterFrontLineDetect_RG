#cmake version
cmake_minimum_required(VERSION 3.2)

#setOpenCV_DIR
SET(OpenCV_DIR "/home/xx/andyoyo/opencv_build/opencv-3.4.4/build")

#project name
PROJECT(regionGrowth)

#find OpenCV
find_package(OpenCV REQUIRED)

#print OpenCV_VERSION on terminal
message(STATUS "OpenCV_VERSION " ${OpenCV_VERSION})

#head file path
INCLUDE_DIRECTORIES(include)

#source directory
AUX_SOURCE_DIRECTORY(src DIR_SRCS)

#add executable file
add_executable(${PROJECT_NAME} ${DIR_SRCS} )

#add link library
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )
