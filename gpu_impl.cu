#include <iostream>
#include <stdint.h>

using std::cout;

inline bool gpuAssert(cudaError_t code) {
	if (code != cudaSuccess) {
		cout << cudaGetErrorString(code) << "\n";
		return false;
	}

	return true;
}

__global__ void adjust_hue_hwc(const int height, const int width,
		uint8_t * const input, uint8_t * const output, const float hue_delta) {

	// multiply by 3 since we're dealing with contiguous RGB bytes for each pixel
	const int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 3;

	// bounds check
	if (idx > height * width * 3) {
		return;
	}

	// RGB to HSV
	const float r = input[idx];
	const float g = input[idx + 1];
	const float b = input[idx + 2];
	const float M = max(r, max(g, b));
	const float m = min(r, min(g, b));
	const float chroma = M - m;

    // v is the same as M
	float h = 0.0, s = 0.0; // v = 0.0;

    // hue
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

    // saturation
	if (M > 0.0) {
		s = chroma / M;

	} else {

		s = 0.0f;
	}


    // hue adjustment
	h = fmod(h + hue_delta, 1.0f); 

	// HSV to RGB
	const float new_h = h * 6.0f;
	const float new_chroma = M * s;
	const float x = chroma * (1.0 - fabs(fmod(new_h, 2.0f) - 1.0f));
	const float new_m = M - chroma;

	const bool between_0_and_1 = new_h >= 0.0 && new_h < 1;
	const bool between_1_and_2 = new_h >= 1.0 && new_h < 2;
	const bool between_2_and_3 = new_h >= 2 && new_h < 3;
	const bool between_3_and_4 = new_h >= 3 && new_h < 4;
	const bool between_4_and_5 = new_h >= 4 && new_h < 5;
	const bool between_5_and_6 = new_h >= 5 && new_h < 6;

    output[idx] = round(new_chroma * (between_0_and_1 || between_5_and_6) +
	  x * (between_1_and_2 || between_4_and_5) + new_m);

	output[idx + 1] = round(new_chroma * (between_1_and_2 || between_2_and_3) +
      x * (between_0_and_1 || between_3_and_4) + new_m);	  

	output[idx + 2] =  round(new_chroma * (between_3_and_4 || between_4_and_5) +
	  x * (between_2_and_3 || between_5_and_6) + new_m);

}

int main(int argc, char **argv) {

	const int h = 352;
	const int w = 352;
	const int total = h * w * 3;

	const int size_bytes = h * w * 3 * sizeof(uint8_t);

	uint8_t * mat_h = (uint8_t *) malloc(size_bytes);
	uint8_t * mat_h2 = (uint8_t *) calloc(h * w * 3, sizeof(uint8_t));
	uint8_t * mat_d = NULL;
	uint8_t * mat_d2 = NULL;

	gpuAssert(cudaMalloc(&mat_d, size_bytes));
	gpuAssert(cudaMalloc(&mat_d2, size_bytes));

	for (int i = 0; i < total; i++) {
		mat_h[i] = abs(rand() % 256);
	}

	gpuAssert(cudaMemcpy(mat_d, mat_h, size_bytes, cudaMemcpyHostToDevice));

	const int threads_per_block = 1024;
	const int blocks = (h * w + (threads_per_block - 1)) / threads_per_block;

    const float hue_delta = 0.0f;

    cudaEvent_t start, end;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // using default stream
    cudaEventRecord(start, 0);

    const int num_invocations = 100;

    for (int i = 0; i < num_invocations; i++) {

    	adjust_hue_hwc<<<blocks, threads_per_block>>>(h, w, mat_d, mat_d2, hue_delta);
    }

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, end);

	gpuAssert(cudaMemcpy(mat_h2, mat_d2, size_bytes, cudaMemcpyDeviceToHost));

	int error_ctr = 0;

	for (int i = 0; i < total; i++) {

		if (mat_h[i] != mat_h2[i]) {

			error_ctr++;
		}
	}

    cout.precision(4);

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

    cout << "\nRGB image size: " << h << "x" << w << "\n";
    cout << "GPU hue_adjust function invocations: " << num_invocations << "\n";
    cout << "Total kernel time: " << elapsed_ms << " ms\n";
    cout << "Per invocation: " << (elapsed_ms / num_invocations) << " ms\n";


	cout << "\nThere were " << error_ctr << " bad pixels out of " << total << "\n";
	cout << "This represents " << (100.0 * error_ctr / total) << "% of pixels\n\n";
	cout << "\n==============================================================\n\n";


	return 0;
}
