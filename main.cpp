#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
// #include "multi_display.cpp"

using namespace cv;
using namespace std;

#define int8 unsigned char

float * rgb_to_hsv(const int rows, const int cols, const int8 * rgb) {

    float *hsv = (float *) calloc(rows * cols * 3, sizeof(float));
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {

            int index = ((row * cols) + col) * 3;
            float r = rgb[index];
            float g = rgb[index + 1];
            float b = rgb[index + 2];
            float M = max(r, max(g, b));
            float m = max(r, min(g, b));
            float chroma = M - m;

            if (chroma > 0) {
                if (M == r) {
                    hsv[index] = fmod((g - b) / chroma, 6.0f);
                } else if (M == g) {
                    hsv[index] = (b - r) / chroma + 2.0;
                } else {
                    hsv[index] = (r - g) / chroma + 4.0;
                }
            }

            hsv[index] /= 6.0;

            if (M > 0) {
                hsv[index + 1] = chroma / M;
            }

            hsv[index + 2] = M;
        }
    }

    return hsv;
}

int8 hsv_to_rgb(const int rows, const int cols, const float * hsv) {

    int8 *rgb = (int8 *) calloc(rows * cols * 3, sizeof(int8));
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {

            int index = ((row * cols) + col) * 3;
            float h = hsv[index] * 6.0;
            float s = hsv[index + 1];
            float v = hsv[index + 2];



        }
    }

//    h = arr[..., 0]
//    s = arr[..., 1]
//    v = arr[..., 2]
//
//    rgb_arr = np.zeros_like(arr)
//
//    chroma = v * s
//
//    h_prime = h * 6.0
//
//    x = chroma * ( 1 - np.abs(h_prime % 2 - 1))
//    m = v - chroma
//
//    rgb_arr[..., 0] = chroma * (
//            np.logical_and(h_prime >= 0, h_prime < 1) +
//            np.logical_and(h_prime >= 5, h_prime < 6)
//    ) + x * (
//            np.logical_and(h_prime >= 1, h_prime < 2) +
//            np.logical_and(h_prime >= 4, h_prime < 5)
//    )
//
//    rgb_arr[..., 1] = chroma * (
//            np.logical_and(h_prime >= 1, h_prime < 3)
//    ) + x * (
//            np.logical_and(h_prime >= 0, h_prime < 1) +
//            np.logical_and(h_prime >= 3, h_prime < 4)
//    )
//
//    rgb_arr[..., 2] = chroma * (
//            np.logical_and(h_prime >= 3, h_prime < 5)
//    ) + x * (
//            np.logical_and(h_prime >= 2, h_prime < 3) +
//            np.logical_and(h_prime >= 5, h_prime < 6)
//    )
//
//    for i in xrange(3):
//    rgb_arr[..., i] += m
//
//            rgb_arr = np.round(rgb_arr * 255.0).astype(np.uint8)
//    rgb_arr = np.minimum(rgb_arr, 255)
//
//    return rgb_arr


}

int main( int argc, char** argv )
{
    string path = "/home/marek/src/rgb_hsv_cuda/spiral.jpg";

    Mat img;
    img = imread(path, CV_LOAD_IMAGE_COLOR);

    int rows = img.rows;
    int cols = img.cols;
    int channels = img.channels();
    int total = rows * cols * channels;

    Mat img2(Size(rows, cols),CV_8UC3);

    cvtColor(img, img2, CV_BGR2RGB);

    cout << "rows = " << rows << ", cols = " << cols << ", channels = " << channels << "\n";

    unsigned char *rgb = img2.data;

    float *hsv = rgb_to_hsv(rows, cols, rgb);

    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", img );                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}