#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iostream>

using namespace std;

#define int8 unsigned char

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
    int64_t seed = 0;
};

//inline float special_fmod(float num, float modulus) {
//    const unsigned char sgn = num < 0.0f;
//    const char sign = pow(-1, sgn);
//    return sgn * modulus + sign * fmod(sign * num, modulus);
//}

float * rgb_to_hsv(const int rows, const int cols, const int8 * rgb) {

    const int total = rows * cols * 3;
    float * hsv = (float *) calloc(total, sizeof(float));

    for (int idx = 0; idx < total; idx += 3) {

        const float r = rgb[idx];
        const float g = rgb[idx + 1];
        const float b = rgb[idx + 2];

        const float M = max(r, max(g, b));
        const float m = min(r, min(g, b));
        const float c = M - m;

        // hue
        if (c > 0.0) {
            if (M == r) {

                const float num = (g - b) / c;
                const unsigned char sgn = num < 0.0f;
                const float sign = pow(-1, sgn);
                hsv[idx] = (sgn * 6.0f + sign * fmod(sign * num, 6.0f)) / 6.0f;  // special_fmod((g - b) / c, 6.0f);

            } else if (M == g) {
                hsv[idx] = ((b - r) / c + 2.0) / 6.0f;
            } else {
                hsv[idx] = ((r - g) / c + 4.0) / 6.0f;
            }
        }

        // saturation
        if (M > 0) {
            hsv[idx + 1] = c / M;
        }

        // value
        hsv[idx + 2] = M;
    }

    return hsv;
}

float * rgb_to_hsv2(const int rows, const int cols, const int8 * rgb) {

    const int total = rows * cols * 3;
    float * const hsv = (float *) calloc(total, sizeof(float));

    const float div6 = 1.0f / 6.0f;

    for (int idx = 0; idx < total; idx += 3) {

        const float r = rgb[idx];
        const float g = rgb[idx + 1];
        const float b = rgb[idx + 2];

        const float M = max(r, max(g, b));
        const float m = min(r, min(g, b));
        const float c = M - m;

        if (c > 0.0f) {

            const float divc = 1.0f / c;

            const float rnum = (g - b) * divc;
            const unsigned char sgn = rnum < 0.0f;
            const float sign = pow(-1, sgn);

            const unsigned char mr = M == r;
            const unsigned char mg = M == g;
            const unsigned char mb = M == b;


            // hue
            hsv[idx] =
                    ((sgn * 6.0f + sign * fmod(sign * rnum, 6.0f)) * div6) * mr +
                     (((b - r) * divc + 2.0) * div6) * (mg && !mr) +
                     (((r - g) * divc + 4.0) * div6) * (mb && !mr && !mg);

        }

        if (M > 0.0f) {

            // saturation
            hsv[idx + 1] = (c / M) * (M > 0.0f);

        }

        // value
        hsv[idx + 2] = M;
    }

    return hsv;
}

int8 * hsv_to_rgb(const int rows, const int cols, const float * hsv) {

    const int total = rows * cols * 3;

    int8 * rgb = (int8 *) calloc(total, sizeof(int8));

    for (int idx = 0; idx < total; idx += 3) {

        const float h = hsv[idx];
        const float s = hsv[idx + 1];
        const float v = hsv[idx + 2];

        const int i = h * 6.0;
        const float f = (h * 6.0) - i;
        const int p = round(v * (1.0 - s));
        const int q = round(v * (1.0 - s * f));
        const int t = round(v * (1.0 - s * (1.0 - f)));

        int r = 0;
        int g = 0;
        int b = 0;

        r =
                v * (i % 6 == 0 || i == 5 || s == 0) +
                q * (i == 1) +
                p * (i == 2 || i == 3) +
                t * (i == 4);

        g = t * (i % 6 == 0) +
            v * (i == 1 || i == 2 || s == 0 && (i % 6 != 0)) +
            q * (i == 3) +
            p * (i == 4 || i == 5);

        b = p * (i % 6 == 0 || i == 1) +
            t * (i == 2) +
            v * (i == 3 || i == 4 || s == 0 && (i % 6 != 0)) +
            q * (i == 5);


//        if (i % 6 == 0) {
//
//            r = v;
//            g = t;
//            b = p;
//
//        } else if (i == 1) {
//
//            r = q;
//            g = v;
//            b = p;
//
//        } else if (i == 2) {
//
//            r = p;
//            g = v;
//            b = t;
//
//        } else if (i == 3) {
//
//            r = p;
//            g = q;
//            b = v;
//
//        } else if (i == 4) {
//
//            r = t;
//            g = p;
//            b = v;
//
//        } else if (i == 5) {
//
//            r = v;
//            g = p;
//            b = q;
//
//        } else if (s == 0) {
//
//            r = v;
//            g = v;
//            b = v;
//
//        }

        rgb[idx] = r;
        rgb[idx + 1] = g;
        rgb[idx + 2] = b;
    }

    return rgb;
}

int main(int argc, char **argv) {

    const int rows = 1300;
    const int cols = 1300;
    const int channels = 3;
    const int total = rows * cols * channels;

    int8 * const rgb = (int8 *) calloc(total, sizeof(int8));

    RNG rng(42);

    for (int i = 0; i < total; i++) {

        rgb[i] = (int8) rng.next_int(255);// (int8) abs(rand() % 255);
    }

    const float * const hsv = rgb_to_hsv2(rows, cols, rgb);
    const int8 * const rgb2 = hsv_to_rgb(rows, cols, hsv);

    const char * const lookup[3]{"red", "green", "blue"};
    int channel_ctr = 0;
    int ctr = 0;

    for (int i = 0; i < total; i++) {

        if (rgb[i] != rgb2[i]) {
           const int diff = abs(rgb[2] - rgb2[i]);

            // diff of 1 can be due to float operations
            if (diff > 1) {
                std::cout << "BAD PIXEL: index " << i << "\n";
                std::cout << "channel = " << lookup[channel_ctr]
                          << ", original = " << (int) rgb[i]
                          << ", after RGB->HSV->RGB = " << (int) rgb2[i]
                          << "\n";
                std::cout << "h pixels before it: [" << (int) rgb[i - 2] << ", " << (int) rgb[i - 1] << "]\n";
                std::cout << "h pixels after it: [" << (int) rgb[i + 1] << ", " << (int) rgb[i + 2] << "]\n\n";
                ctr++;
            }
        }

        channel_ctr = (channel_ctr + 1) % 3;
    }

    cout.precision(4);
    cout << "\nPercent bad pixels: " << (1.0 * ctr / total * 100) << "\n";

//    cout << "\n Number of inequalities: " << counter << "\n";
//    cout << "\n Number of inequalities > 1: " << ctr2 << "\n";
//    cout << "Percentage of total (> 1): " << (ctr2 * 1.0 / total * 100) << "\n";
//    cout << "Max inequality: " << maxerr << "\n";

//    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
//    imshow( "Display window", img );                   // Show our image inside it.
//
//    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
