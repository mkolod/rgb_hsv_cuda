#include <stdint.h> 

__global__ void adjust_hue_hwc(const int height, const int width,
		uint8_t * const __restrict__ input, uint8_t * const __restrict__ output, const float hue_delta) {

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
	const float M = fmaxf(r, fmaxf(g, b));
	const float m = fminf(r, fminf(g, b));
	const float chroma = M - m;

    // v is the same as M
	float h = 0.0f, s = 0.0f; // v = 0.0;

    // hue
	if (chroma > 0.0f) {
		if (M == r) {

            const float num = (g - b) / chroma;
            const float sgn = num < 0.0f;
            const float sign = powf(-1.0f, sgn);
            h = (sgn * 6.0f + sign * fmodf(sign * num, 6.0f)) / 6.0f;
			
		} else if (M == g) {

			h = ((b - r) / chroma + 2.0f) / 6.0f;

		} else {

			h = ((r - g) / chroma + 4.0f) / 6.0f;
		}        

	} else {

		h = 0.0f;
	}

    // saturation
	if (M > 0.0f) {
		s = chroma / M;

	} else {

		s = 0.0f;
	}


    // hue adjustment
	h = fmodf(h + hue_delta, 1.0f); 

   ////////////////////////////////////////////
   // Murmurhash - based random adjustment
   // uint32_t k = idx;
   // uint32_t seed = 42;

   // k *= 0xcc9e2d51;
   // k = (k << 15) | (k >> 17);
   // k *= 0x1b873593;
   // seed ^= k;

   // seed ^= 1;
   // seed ^= seed >> 16;
   // seed *= 0x85ebca6b;
   // seed ^= seed >> 13;
   // seed *= 0xc2b2ae35;
   // seed ^= seed >> 16;

   // // TODO: Add scaling factor
   // // TODO: Cover shifts both up and down with wrap-around
   // float rand_delta = (seed >> 8) / (float) (1 << 24);
   // h = fmod(h + rand_delta, 1.0f);

   ////////////////////////////////////////////




	// HSV to RGB
	const float new_h = h * 6.0f;
	const float new_chroma = M * s;
	const float x = chroma * (1.0 - fabs(fmod(new_h, 2.0f) - 1.0f));
	const float new_m = M - chroma;

	const bool between_0_and_1 = new_h >= 0.0f && new_h < 1.0f;
	const bool between_1_and_2 = new_h >= 1.0f && new_h < 2.0f;
	const bool between_2_and_3 = new_h >= 2.0f && new_h < 3.0f;
	const bool between_3_and_4 = new_h >= 3.0f && new_h < 4.0f;
	const bool between_4_and_5 = new_h >= 4.0f && new_h < 5.0f;
	const bool between_5_and_6 = new_h >= 5.0f && new_h < 6.0f;

    output[idx] = roundf(new_chroma * (between_0_and_1 || between_5_and_6) +
	  x * (between_1_and_2 || between_4_and_5) + new_m);

	output[idx + 1] = roundf(new_chroma * (between_1_and_2 || between_2_and_3) +
      x * (between_0_and_1 || between_3_and_4) + new_m);	  

	output[idx + 2] =  roundf(new_chroma * (between_3_and_4 || between_4_and_5) +
	  x * (between_2_and_3 || between_5_and_6) + new_m);

}
