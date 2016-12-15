cmake_minimum_required(VERSION 3.5)

find_package(CUDA)
find_package(OpenCV REQUIRED)

file(GLOB  hdr *.hpp *.h)
file(GLOB  cu  *.cu)
SET (project_files main.cpp multi_display.cpp)


add_executable( DisplayImage main.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )

# CUDA_ADD_EXECUTABLE(HW1 ${HW1_files} ${hdr} ${cu})
