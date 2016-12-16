#include <stdint.h>

#ifndef __ADJUST_HUE_CUDA
#define __ADJUST_HUE_CUDA

__global__ void adjust_hue_hwc(const int height, const int width,
		uint8_t * const input, uint8_t * const output, const float hue_delta);

#endif
