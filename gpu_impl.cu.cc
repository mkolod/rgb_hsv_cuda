#include <stdint.h> 

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