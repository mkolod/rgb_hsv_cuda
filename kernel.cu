#define int8 unsigned char

// #include <cuda.h>
// #include <math_functions.hpp>
// #include <cuda_runtime.h>
// #include <cstdlib>
#include <iostream>

// #define __CUDA_INTERNAL_COMPILATION__
// #include <math_functions.h>
// #undef __CUDA_INTERNAL_COMPILATION__

using std::cout;

inline bool gpuAssert(cudaError_t code) {
	if (code != cudaSuccess) {
		cout << cudaGetErrorString(code) << "\n";
		return false;
	}

	return true;
}

__global__ void adjust_hue_hwc(const int height, const int width,
		int8 * const input, int8 * const output) {

	// multiply by 3 since we're dealing with contiguous RGB bytes for each pixel
	const int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 3;

	// bounds check
	if (idx > height * width * 3) {
		return;
	}

	// rgb_to_hsv
	const float r = input[idx];
	const float g = input[idx + 1];
	const float b = input[idx + 2];
	const float M = max(r, max(g, b));
	const float m = min(r, min(g, b));
	const float chroma = M - m;

    // not allocating space for v because v = M by definition
	float h = 0.0, s = 0.0;

	if (chroma > 0.0f) {
		if (M == r) {

            const float num = (g - b) / chroma;
            const float sgn = num < 0.0f;
            const float sign = pow(-1.0f, sgn);
            h = (sgn * 6.0f + sign * fmod(sign * num, 6.0f)) / 6.0f;
			
		} else if (M == g) {

			h = ((b - r) / chroma + 2.0) / 6.0f;

		} else {

			h = ((r - g) / chroma + 4.0) / 6.0f;
		}

	} else {

		h = 0.0f;
	}

	if (M > 0.0) {

		s = chroma / M;

	} else {

		s = 0.0f;
	}

    // we need multiplication, then truncation
    const int i = h * 6.0;
    const float f = (h * 6.0) - i;
    const int p = round(M * (1.0 - s));
    const int q = round(M * (1.0 - s * f));
    const int t = round(M * (1.0 - s * (1.0 - f)));

    output[idx] =
            M * (i % 6 == 0 || i == 5 || s == 0) +
            q * (i == 1) +
            p * (i == 2 || i == 3) +
            t * (i == 4);

    output[idx + 1] = t * (i % 6 == 0) +
        M * (i == 1 || i == 2 || s == 0 && (i % 6 != 0)) +
        q * (i == 3) +
        p * (i == 4 || i == 5);

    output[idx + 2] = p * (i % 6 == 0 || i == 1) +
        t * (i == 2) +
        M * (i == 3 || i == 4 || s == 0 && (i % 6 != 0)) +
        q * (i == 5);
}

int main(void) {

	srand(1);

	const int h = 352; //1300;
	const int w = 352; //1300;
	const int total = h * w * 3;

	const int size_bytes = h * w * 3 * sizeof(int8);

	int8 * mat_h = (int8 *) malloc(size_bytes);
	int8 * mat_h2 = (int8 *) calloc(h * w * 3, sizeof(int8));
	int8 * mat_d = NULL;
	int8 * mat_d2 = NULL;

	gpuAssert(cudaMalloc(&mat_d, size_bytes));
	gpuAssert(cudaMalloc(&mat_d2, size_bytes));

	for (int i = 0; i < total; i++) {
		const int num = rand() % 256;
		mat_h[i] = num > 0.0f ? num : -num;
	}

	gpuAssert(cudaMemcpy(mat_d, mat_h, size_bytes, cudaMemcpyHostToDevice));

	const int threads_per_block = 1024;
	const int blocks = (h * w + (threads_per_block - 1)) / threads_per_block;

    for (int i = 0; i < 100; i++) {
    	adjust_hue_hwc<<<blocks, threads_per_block>>>(h, w, mat_d, mat_d2);
    }

	gpuAssert(cudaMemcpy(mat_h2, mat_d2, size_bytes, cudaMemcpyDeviceToHost));

	int error_ctr = 0;
	int channel_ctr = 0;

	using std::cout;
	// const char * lookup[3] { "red", "green", "blue" };

	for (int i = 0; i < total; i++) {
		channel_ctr = (channel_ctr + 1) % 3;
		if (abs(mat_h[i] - mat_h2[i]) > 0) {
//			std::cout << "BAD PIXEL: index " << i << "\n";
//			std::cout << "channel = " << lookup[channel_ctr]
//					<< ", original = " << (int) mat_h[i]
//					<< ", after GPU RGB->HSV->RGB = " << (int) mat_h2[i]
//					<< "\n";
//			std::cout<< "h pixels before it: [" << (int) mat_h[i - 2] << ", " << (int) mat_h[i - 1] << "]\n";
//			std::cout<< "h pixels after it: [" << (int)  mat_h[i + 1] << ", " << (int)  mat_h[i + 2] << "]\n\n";
			error_ctr++;
		}
	}


	cout << "\nThere were " << error_ctr << " bad pixels out of " << total << "\n";
	cout << "This represents " << (100.0 * error_ctr / total) << "%\n\n";

	return 0;
}
