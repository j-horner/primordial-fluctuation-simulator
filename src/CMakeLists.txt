#
#   Target names are global so if this project is included as a subdirectory of another, name clashes can occur.
#   Therefore consider keeping target names project specific.
#
add_executable(${PROJECT_NAME}_exe oceanFFT.cpp oceanFFT_kernel.cu)

# add_library(${PROJECT_NAME}_kernel STATIC oceanFFT_kernel.cu)

# target_compile_features(${PROJECT_NAME}_exe PUBLIC cxx_std_20)


# target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=sm_70 -Werror cross-execution-space-call,deprecated-declarations,reorder -expt-extended-lambda -expt-relaxed-constexpr>)
target_compile_options(${PROJECT_NAME}_exe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Werror cross-execution-space-call,deprecated-declarations,reorder -expt-extended-lambda -expt-relaxed-constexpr>)

# We need to explicitly state that we need all CUDA files in the
# gpu library to be built with -dc as the member functions
# could be called by other libraries and executables
set_target_properties(${PROJECT_NAME}_exe PROPERTIES  CUDA_SEPARABLE_COMPILATION ON
                                                         POSITION_INDEPENDENT_CODE ON
                                                         CUDA_RESOLVE_DEVICE_SYMBOLS ON
                                                         CUDA_STANDARD 20)


#
#   find_package() will import an appropriately namespaced target
#
add_executable(${PROJECT_NAME}::exe ALIAS ${PROJECT_NAME}_exe)
set_target_properties(${PROJECT_NAME}_exe PROPERTIES OUTPUT_NAME ${PROJECT_NAME} EXPORT_NAME exe)

target_link_libraries(${PROJECT_NAME}_exe
                        PRIVATE
                            CUDA::cudart_static
                            CUDA::curand
                            CUDA::cufft
                            OpenGL::GL
                            GLUT::GLUT
                            GLEW::GLEW)
