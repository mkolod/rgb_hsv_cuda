#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <ctime>
#include <stdint.h>

using namespace std;

#define int8 unsigned char

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


void hue_adjust(const int rows, const int cols, const int8 * const rgb, int8 * const rgb2, const float hue_delta) {

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

    const int rows = 352; //1300;
    const int cols = 352; //1300;
    const int channels = 3;
    const int total = rows * cols * channels;

    int8 * const rgb = (int8 *) calloc(total, sizeof(int8));

    RNG rng(42);

    for (int i = 0; i < total; i++) {

        rgb[i] = (int8) rng.next_int(255);// (int8) abs(rand() % 255);
    }
    
    int8 * const rgb2 = (int8 *) calloc(total, sizeof(int8));

    std::clock_t start, end;
    start = clock();

    const float hue_delta = 0.0f;

    const int num_invocations = 100;

    for (int i = 0; i < num_invocations; i++) {
        hue_adjust(rows, cols, rgb, rgb2, hue_delta);
    }

    end = clock();

    const float total_time = 1000.0 * (end - start) / (CLOCKS_PER_SEC);

    cout.precision(4);

    cout << "\nCPU implementation\n";
    cout << "\nRGB image size: " << rows << "x" << cols << "\n";
    cout << "CPU hue_adjust function invocations: " << num_invocations << "\n";
    cout << "Total kernel time: " << total_time << " ms\n";
    cout << "Per invocation: " << (total_time / num_invocations) << " ms\n";

    int ctr = 0;

    for (int i = 0; i < total; i++) {

        if (rgb[i] != rgb2[i]) {

            ctr++;
        }

    }

    cout << "\nPercent bad pixels: " << (1.0 * ctr / total * 100) << "\n\n";

//    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
//    imshow( "Display window", img );                   // Show our image inside it.
//
//    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
