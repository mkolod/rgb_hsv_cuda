export CFLAGS="-O3 -march=native -mtune=native -m64 -ffast_math"

nvcc -O3 -c -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61 gpu_impl.cu -o gpu_impl.o

nvcc -O3 -arch=sm_61 cpu_impl.cu gpu_impl.o -o cpu_impl `pkg-config opencv --cflags --libs`

./cpu_impl

