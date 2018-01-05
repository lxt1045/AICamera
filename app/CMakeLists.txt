cmake_minimum_required(VERSION 3.4.1)

add_library(
             native-lib
             SHARED
             src/main/cpp/native-lib.cpp
             )
find_library(
          android-lib
          android
          )

include(AndroidNdkModules)
android_ndk_import_module_cpufeatures()

add_library(
    caffe2
    STATIC
    IMPORTED
    )
set_target_properties(
    caffe2
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libCaffe2_CPU.a
    )
add_library(
    thread_pool
    STATIC
    IMPORTED
    )
set_target_properties(
    thread_pool
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libCAFFE2_PTHREADPOOL.a
    )
add_library(
    glog
    SHARED
    IMPORTED
    )
set_target_properties(
    glog
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libglog.so
    )

add_library(
    protobuf
    SHARED
    IMPORTED
    )
set_target_properties(
    protobuf
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libprotobuf.a
    )

add_library(
    NNPACK
    STATIC
    IMPORTED
    )
set_target_properties(
    NNPACK
    PROPERTIES IMPORTED_LOCATION
    ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libCAFFE2_NNPACK.a
    )

set(ocvlibs "${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs")
# add_library(
#     opencv_core
#     STATIC
#     IMPORTED
#     )
# set_target_properties(
#     opencv_core
#     PROPERTIES IMPORTED_LOCATION
#     ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopencv_core.a
#     )
# add_library(
#     opencv_highgui
#     STATIC
#     IMPORTED
#     )
# set_target_properties(
#     opencv_highgui
#     PROPERTIES IMPORTED_LOCATION
#     ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libopencv_highgui.a
#     )

# add_library(
#     libtegra_hal
#     STATIC
#     IMPORTED
#     )
# set_target_properties(
#     libtegra_hal
#     PROPERTIES IMPORTED_LOCATION
#     ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libtegra_hal.a
#     )
add_library(
    libopencv_java3 
    SHARED
    IMPORTED 
    )
set_target_properties(
    libopencv_java3 
    PROPERTIES IMPORTED_LOCATION 
    "${ocvlibs}/${ANDROID_ABI}/libopencv_java3.so")

include_directories( 
    src/main/cpp 
    src/main/include
    #D:/Program Files/Android/OpenCV-android-sdk/sdk/native/jni/include
    )

find_library(
     log-lib
     log
     )

target_link_libraries(
                       native-lib
                       libopencv_java3
                       -Wl,--whole-archive
                    #    opencv_core
                    #    opencv_highgui
                    #    libtegra_hal
                       caffe2
                       -Wl,--no-whole-archive
                       NNPACK
                       thread_pool
                       glog
                       protobuf
                       cpufeatures
                       ${log-lib}
                       ${android-lib})