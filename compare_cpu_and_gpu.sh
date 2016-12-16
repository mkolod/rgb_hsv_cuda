nvcc -O3 -c -arch=sm_61 gpu_impl.cu -o gpu_impl.o
nvcc -O3 -arch=sm_61 cpu_impl.cu gpu_impl.o -o cpu_impl
./cpu_impl

