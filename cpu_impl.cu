#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <ctime>
#include <stdint.h>
#include <cpuid.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "gpu_impl.cuh"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using cv::Mat;
using cv::imread;
using cv::imshow;
using cv::namedWindow;
using cv::WINDOW_AUTOSIZE;
using cv::waitKey;

using namespace std;

inline bool gpuAssert(cudaError_t code) {
	if (code != cudaSuccess) {
		cout << cudaGetErrorString(code) << "\n";
		return false;
	}

	return true;
}

// RNG for reproducible CPU/GPU results without loading an actual RGB image
class RNG {
public:

    RNG(const uint64_t seed) : seed(seed) {}

    uint32_t next_int(const uint32_t max) {
        uint64_t next = (1664525 * seed + 1013904223) % (2L << 32);
        seed = next;
        if (next > max) {
            next %= max;
        }
        return next;
    }

private:
    int64_t seed;
};

// For CPUID
enum cpuid_requests {
  CPUID_INTELEXTENDED=0x80000000,
  CPUID_INTELFEATURES,
  CPUID_INTELBRANDSTRING,
  CPUID_INTELBRANDSTRINGMORE,
  CPUID_INTELBRANDSTRINGEND,
};

static inline void cpuid(int code, int *a, int *b, int *c, int *d) {
  __asm__ __volatile__("cpuid":"=a"(*a),"=b"(*b),
                        "=c"(*c),"=d"(*d):"a"(code));
}

void print_cpu_id() {

	union{
     struct reg{
         int eax;
         int ebx;
         int ecx;
         int edx;
     }cpu;
   char string[16];
  }info;

  cout << "Processor Brand:  ";

  cpuid(CPUID_INTELBRANDSTRING, &info.cpu.eax, &info.cpu.ebx, &info.cpu.ecx, &info.cpu.edx);
  cout << std::string(info.string, 16);

  cpuid(CPUID_INTELBRANDSTRINGMORE, &info.cpu.eax, &info.cpu.ebx, &info.cpu.ecx, &info.cpu.edx);
  cout << std::string(info.string, 16);

  cpuid(CPUID_INTELBRANDSTRINGEND, &info.cpu.eax, &info.cpu.ebx, &info.cpu.ecx, &info.cpu.edx);
  cout << std::string(info.string, 16)  << "\n";
}


void hue_adjust(const int rows, const int cols, const uint8_t * const rgb, uint8_t * rgb2, const float hue_delta) {

    const int total = rows * cols * 3;   

    for (int idx = 0; idx < total; idx += 3) {

        // RGB to HSV

        const float r = rgb[idx];
        const float g = rgb[idx + 1];
        const float b = rgb[idx + 2];

        const float M = max(r, max(g, b));
        const float m = min(r, min(g, b));
        const float c = M - m;

        // value is the same as M
        float h = 0.0f;
        float s = 0.0f;

        // hue
        if (c > 0.0) {

            if (M == r) {

                const float num = (g - b) / c;
                const unsigned char sgn = num < 0.0f;
                const float sign = pow(-1, sgn);
                h = (sgn * 6.0f + sign * fmod(sign * num, 6.0f)) / 6.0f;

            } else if (M == g) {

                h = ((b - r) / c + 2.0) / 6.0f;

            } else {

                h = ((r - g) / c + 4.0) / 6.0f;
            }            

        }        

        // saturation
        if (M > 0) {
            s = c / M;
        }

        // hue adjustment
        h = fmod(h + hue_delta, 1.0f);

        // HSV to RGB
        const int i = h * 6.0;
        const float f = (h * 6.0) - i;
        const int p = round(M * (1.0 - s));
        const int q = round(M * (1.0 - s * f));
        const int t = round(M * (1.0 - s * (1.0 - f)));

        rgb2[idx] =
                M * (i % 6 == 0 || i == 5 || s == 0) +
                q * (i == 1) +
                p * (i == 2 || i == 3) +
                t * (i == 4);

        rgb2[idx + 1] = t * (i % 6 == 0) +
            M * (i == 1 || i == 2 || s == 0 && (i % 6 != 0)) +
            q * (i == 3) +
            p * (i == 4 || i == 5);

        rgb2[idx + 2] = p * (i % 6 == 0 || i == 1) +
            t * (i == 2) +
            M * (i == 3 || i == 4 || s == 0 && (i % 6 != 0)) +
            q * (i == 5);
    
    }    

}


int main(int argc, char **argv) {

    const int rows = 352;
    const int cols = 352;
    const int channels = 3;
    const int total = rows * cols * channels;

    uint8_t * const rgb = (uint8_t *) calloc(total, sizeof(uint8_t));

    RNG rng(42);

    for (int i = 0; i < total; i++) {

        rgb[i] = (uint8_t) rng.next_int(255);
    }
    
    uint8_t * const rgb2 = (uint8_t *) calloc(total, sizeof(uint8_t));

    std::clock_t start, end;
    start = clock();

    const float hue_delta = 0.0f;

    const int num_invocations = 100;

    for (int i = 0; i < num_invocations; i++) {
        hue_adjust(rows, cols, rgb, rgb2, hue_delta);
    }

    end = clock();

    const float total_cpu_time = 1000.0 * (end - start) / (CLOCKS_PER_SEC);

    cout.precision(4);

    cout << "\n==============================================================\n";
    cout << "\nHue adjustment - CPU implementation\n";
    cout << "CPU Info: " << std::flush;    
    // Note: this will only work on Intel x86 and clones (e.g. AMD)
    print_cpu_id();
    cout << "RGB image size: " << rows << "x" << cols << "\n";
    cout << "CPU hue_adjust function invocations: " << num_invocations << "\n";
    cout << "Total hue_adjust function time: " << total_cpu_time << " ms\n";
    cout << "Per invocation: " << (total_cpu_time / num_invocations) << " ms\n";

    int error_ctr = 0;

    for (int i = 0; i < total; i++) {

        if (rgb[i] != rgb2[i]) {

            error_ctr++;
        }

    }

    cout << "\nThere were " << error_ctr << " bad pixels out of " << total << "\n";
    cout << "This represents " << (100.0 * error_ctr / total) << "% of pixels\n\n";    

   	uint8_t * rgb_h2 = (uint8_t *) calloc(rows * cols * 3, sizeof(uint8_t));
	uint8_t * rgb_d = NULL;
	uint8_t * rgb_d2 = NULL;

    const int size_bytes = total * sizeof(uint8_t);

	gpuAssert(cudaMalloc(&rgb_d, size_bytes));
	gpuAssert(cudaMalloc(&rgb_d2, size_bytes));

	gpuAssert(cudaMemcpy(rgb_d, rgb, size_bytes, cudaMemcpyHostToDevice));

	const int threads_per_block = 1024;
	const int blocks = (rows * cols + (threads_per_block - 1)) / threads_per_block;

	cudaEvent_t cuda_start, cuda_end;

    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_end);

    // using default stream
    cudaEventRecord(cuda_start, 0);

    for (int i = 0; i < num_invocations; i++) {

    	adjust_hue_hwc<<<blocks, threads_per_block>>>(rows, cols, rgb_d, rgb_d2, hue_delta);
    }

    cudaEventRecord(cuda_end, 0);
    cudaEventSynchronize(cuda_end);

    float total_gpu_time;
    cudaEventElapsedTime(&total_gpu_time, cuda_start, cuda_end);

	gpuAssert(cudaMemcpy(rgb_h2, rgb_d2, size_bytes, cudaMemcpyDeviceToHost));

	error_ctr = 0;

	for (int i = 0; i < total; i++) {

		if (rgb[i] != rgb_h2[i]) {

			error_ctr++;
		}
	}

    cout << "\n==============================================================\n";
    cout << "\nHue adjustment - GPU implementation\n";

    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    cout << "GPU: " << deviceProp.name << "\n";
    cout << "CUDA driver version : " << (driverVersion / 1000) << "\n";
    cout << "CUDA runtime version : " << (runtimeVersion / 1000) << "\n";
    cout << "CUDA capability major.minor: " << deviceProp.major << "." << deviceProp.minor << "\n";

    cout << "\nRGB image size: " << rows << "x" << cols << "\n";
    cout << "GPU hue_adjust function invocations: " << num_invocations << "\n";
    cout << "Total kernel time: " << total_gpu_time << " ms\n";
    cout << "Per invocation: " << (total_gpu_time / num_invocations) << " ms\n";


	cout << "\nThere were " << error_ctr << " bad pixels out of " << total << "\n";
	cout << "This represents " << (100.0 * error_ctr / total) << "% of pixels\n\n";
	cout << "\n==============================================================\n\n";


    cout << "GPU speed-up over CPU: " << (total_cpu_time / total_gpu_time) << "x\n\n";

    cout << "Running OpenCV test with hue adjustment applied\n\n";

    Mat img;
    img = imread("lenna.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file

   Mat img2 = img.clone();
   uint8_t * h_origdata = img.data;
   uint8_t * h_newdata = (uint8_t *) calloc(img2.rows * img2.cols * 3, sizeof(uint8_t));

   uint8_t * d_origdata = NULL;
   uint8_t * d_newdata = NULL;

   const int new_size_bytes = img2.rows * img2.cols * 3 * sizeof(uint8_t);

   gpuAssert(cudaMalloc(&d_origdata, new_size_bytes));
   gpuAssert(cudaMalloc(&d_newdata, new_size_bytes));

   gpuAssert(cudaMemcpy(d_origdata, h_origdata, new_size_bytes, cudaMemcpyHostToDevice));

   const int new_blocks = (img2.rows * img2.cols + (threads_per_block - 1)) / threads_per_block;

   const float new_hue_delta = 0.7;

   adjust_hue_hwc<<<new_blocks, threads_per_block>>>(img2.rows, img2.cols, d_origdata, d_newdata, new_hue_delta);

   gpuAssert(cudaMemcpy(h_newdata, d_newdata, new_size_bytes, cudaMemcpyDeviceToHost));

   img2.data = h_newdata;

   Mat img3 = img.clone();
   hue_adjust(img3.rows, img3.cols, img.data, img3.data, 0.7); // -0.6

   Mat side_by_side(img2.rows * 3, img2.cols * 3, CV_8UC3);

   // Mat * ptr = (Mat *) malloc(2 * sizeof(Mat));
   Mat arr[3];
   arr[0] = img;
   arr[1] = img2;
   arr[2] = img3;

   cv::hconcat(arr, 3, side_by_side);

   namedWindow("Display window", WINDOW_AUTOSIZE);
   imshow("Display window", side_by_side);   


//   waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
