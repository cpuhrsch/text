cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)


add_library(torchtext SHARED IMPORTED GLOBAL)
set_target_properties(torchtext PROPERTIES IMPORTED_LOCATION "/private/home/cpuhrsch/repos/text/build/lib.linux-x86_64-3.8/torchtext/_torchtext.so")
# target_link_libraries(TorchText::TorchText 
# "/private/home/cpuhrsch/repos/text/build/lib.linux-x86_64-3.8/torchtext/_torchtext.so")

message("${TORCH_LIBRARIES}")
add_executable(benchmark-jit_pipeline benchmark_jit_pipeline.cpp)
target_link_libraries(benchmark-jit_pipeline torchtext)
target_link_libraries(benchmark-jit_pipeline -Wl,--no-as-needed torchtext)
target_link_libraries(benchmark-jit_pipeline "${TORCH_LIBRARIES}")
# set_property(TARGET benchmark-jit_pipeline PROPERTY CXX_STANDARD 14)
