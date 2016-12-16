g++ -O3 cpu_impl.cpp -o cpu_impl && ./cpu_impl

nvcc -O3 -arch=sm_61 gpu_impl.cu -o gpu_impl && ./gpu_impl
