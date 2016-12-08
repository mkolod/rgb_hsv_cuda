#define int8 unsigned char

#include <cstdlib>
#include <iostream>

using std::cout;

inline void gpuAssert(cudaError_t code) {
	if (code != cudaSuccess) {
		std::cout << cudaGetErrorString(code) << "\n";
	}
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

	float h = 0.0, s = 0.0, v = 0.0;

	if (chroma > 0) {
		if (M == r) {
			h = fmod((g - b) / chroma, 6.0f);
		} else if (M == g) {
			h = (b - r) / chroma + 2.0;
		} else {
			h = (r - g) / chroma + 4.0;
		}
	}

	if (M > 0.0) {
		s = chroma / M;
	}

	v = M;


	// hsv2rgb
	const float new_chroma = v * s;
	const float x = chroma * (1.0 - fabs(fmod(h, 2.0f) - 1.0f));
	const float new_m = v - chroma;

	const int between_0_and_1 = h >= 0.0 && h < 1;
	const int between_1_and_2 = h >= 1.0 && h < 2;
	const int between_2_and_3 = h >= 2 && h < 3;
	const int between_3_and_4 = h >= 3 && h < 4;
	const int between_4_and_5 = h >= 4 && h < 5;
	const int between_5_and_6 = h >= 5 && h < 6;

	// red channel
	const int red_chroma_mask = between_0_and_1 || between_5_and_6;
	const int red_x_mask = between_1_and_2 || between_4_and_5;

	const int8 new_r = new_chroma * red_chroma_mask + x * red_x_mask + new_m;

	// green channel
	const int green_chroma_mask = between_1_and_2 || between_2_and_3;
	const int green_x_mask = between_0_and_1 || between_3_and_4;

	const int8 new_g = new_chroma * green_chroma_mask + x * green_x_mask
			+ new_m;

	// blue channel
	const int blue_chroma_mask = between_3_and_4 || between_4_and_5;
	const int blue_x_mask = between_2_and_3 || between_5_and_6;

	const int8 new_b = new_chroma * blue_chroma_mask + x * blue_x_mask + new_m;

	output[idx] = new_r;
	output[idx + 1] = new_g;
	output[idx + 2] = new_b;

}

int main(void) {

	srand(1);

	const int h = 1300;
	const int w = 1300;
	const int total = h * w * 3;

	const int size_bytes = h * w * 3 * sizeof(int8);

	int8 * mat_h = (int8 *) malloc(size_bytes);
	int8 * mat_h2 = (int8 *) calloc(h * w * 3, sizeof(int8));
	int8 * mat_d = NULL;
	int8 * mat_d2 = NULL;

	gpuAssert(cudaMalloc(&mat_d, size_bytes));
	gpuAssert(cudaMalloc(&mat_d2, size_bytes));

	for (int i = 0; i < total; i++) {
		mat_h[i] = abs(rand() % 256);
	}

	gpuAssert(cudaMemcpy(mat_d, mat_h, size_bytes, cudaMemcpyHostToDevice));

	const int threads_per_block = 1024;
	const int blocks = (h * w + (threads_per_block - 1)) / threads_per_block;

	adjust_hue_hwc<<<blocks, threads_per_block>>>(h, w, mat_d, mat_d2);

	gpuAssert(cudaMemcpy(mat_h2, mat_d2, size_bytes, cudaMemcpyDeviceToHost));

	int error_ctr = 0;
	int channel_ctr = 0;

	using std::cout;
	const char * lookup[3] { "red", "green", "blue" };

	for (int i = 0; i < total; i++) {
		channel_ctr = (channel_ctr + 1) % 3;
		if (abs(mat_h[i] - mat_h2[i]) > 1) {
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
