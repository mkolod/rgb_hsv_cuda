//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <iostream>
//// #include "multi_display.cpp"
//
//using namespace cv;
//using namespace std;
//
//#define int8 unsigned char
//
//
//__global__ void adjust_hue_hwc(const int h, const int w, const int8 * input, const int8 * output) {
//
//	// multiply by 3 since we're dealing with contiguous RGB bytes for each pixel
//	const int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 3;
//
//	// bounds check
//	if (idx > h * w * 3) {
//		return;
//	}
//
//	// rgb_to_hsv
//	float r = input[idx];
//	float g = input[idx + 1];
//	float b = input[idx + 2];
//	float M = max(r, max(g, b));
//
//
//}
//
//float * rgb_to_hsv(const int rows, const int cols, const int8 * rgb) {
//
//    float *hsv = (float *) calloc(rows * cols * 3, sizeof(float));
//    for (int row = 0; row < rows; row++) {
//        for (int col = 0; col < cols; col++) {
//
//            int index = ((row * cols) + col) * 3;
//            float r = rgb[index];
//            float g = rgb[index + 1];
//            float b = rgb[index + 2];
//            float M = max(r, max(g, b));
//            float m = min(r, min(g, b));
//            float chroma = M - m;
//
//            // hue
//            if (chroma > 0) {
//                if (M == r) {
//                    hsv[index] = fmod((g - b) / chroma, 6.0f);
//                } else if (M == g) {
//                    hsv[index] = (b - r) / chroma + 2.0;
//                } else {
//                    hsv[index] = (r - g) / chroma + 4.0;
//                }
//            }
//
//            hsv[index] /= 6.0;
//
//            // saturation
//            if (M > 0) {
//                hsv[index + 1] = chroma / M;
//            }
//
//            // value
//            hsv[index + 2] = M;
//        }
//    }
//
//    return hsv;
//}
//
//
//int8 * hsv_to_rgb(const int rows, const int cols, const float * hsv) {
//
//    int8 *rgb = (int8 *) calloc(rows * cols * 3, sizeof(int8));
//    for (int row = 0; row < rows; row++) {
//        for (int col = 0; col < cols; col++) {
//
//            const int index = ((row * cols) + col) * 3;
//            float h = hsv[index] * 6.0;
//            float s = hsv[index + 1];
//            float v = hsv[index + 2];
//
//            float chroma = v * s;
//            float x = chroma * (1.0 - fabs(fmod(h, 2.0f) - 1.0f));
//            float m = v - chroma;
//
//
//            rgb[index] = chroma * (
//            		(h >= 0 && h < 1) || (h >= 5 && h < 6)
//            		) + x * (
//            				(h >= 1 && h < 2) || (h >= 4 && h < 5)
//            				) + m;
//
//            rgb[index + 1] = chroma * (
//            		(h >= 1 && h < 3)
//            		) + x * (
//            				(h >= 0 && h < 1) || (h >= 3 && h < 4)
//            				) + m;
//
//            rgb[index + 2] = chroma * (
//            		(h >= 3 && h < 5)
//            		) + x * (
//            				(h >= 2 && h < 3) || (h >= 5 && h < 6)
//            				) + m;
//
//
////            int between_0_and_1 = h >= 0.0 && h < 1;
////            int between_1_and_2 = h >= 1.0 && h < 2;
////            int between_2_and_3 = h >= 2 && h < 3;
////            int between_3_and_4 = h >= 3 && h < 4;
////            int between_4_and_5 = h >= 4 && h < 5;
////            int between_5_and_6 = h >= 5 && h < 6;
////
////            // red channel
////            int red_chroma_mask = between_0_and_1 || between_5_and_6;
////            int red_x_mask = between_1_and_2 || between_4_and_5;
////
////            rgb[index] = chroma * red_chroma_mask + x * red_x_mask + m;
////
////            // green channel
////            int green_chroma_mask = between_1_and_2 || between_2_and_3;
////            int green_x_mask = between_0_and_1 || between_3_and_4;
////
////            rgb[index + 1] = chroma * green_chroma_mask + x * green_x_mask + m;
////
////            // blue channel
////            int blue_chroma_mask = between_3_and_4 || between_4_and_5;
////            int blue_x_mask = between_2_and_3 || between_5_and_6;
////
////            rgb[index + 2] = chroma * blue_chroma_mask + x * blue_x_mask + m;
//
//            // clip at 255
//
//        }
//    }
//
//    return rgb;
//}
//
//int main( int argc, char** argv )
//{
//    string path = "spiral.jpg";
//
//    Mat img;
//    img = imread(path, CV_LOAD_IMAGE_COLOR);
//
//    int rows = img.rows;
//    int cols = img.cols;
//    int channels = img.channels();
//    int total = rows * cols * channels;
//
//    Mat img2(Size(rows, cols), CV_8UC3);
//
//    cvtColor(img, img2, CV_BGR2RGB);
//
//    cout << "rows = " << rows << ", cols = " << cols << ", channels = " << channels << "\n";
//
//    unsigned char *rgb = img2.data;
//
//    float *hsv = rgb_to_hsv(rows, cols, rgb);
//    int8 *rgb2 = hsv_to_rgb(rows, cols, hsv);
//
////    for (int i = 0; i < 10000; i++) {
////    	cout << "i = " << i << ", rgb[i] = " << (int) rgb[i] << ", rgb2[i] = " << (int) rgb2[i] << "\n";
////    }
//
//    int counter = 0;
//    int ctr2 = 0;
//    int maxerr = 0;
////
//    for (int i = 0; i < total; i++) {
//    	if (rgb[i] == rgb2[i]) {
////    		cout << "i = " << i << ", rgb[i] = " << (int) rgb[i] << ", rgb2[i] = " << (int) rgb2[i] << "\n";
//    	} else {
////    		cerr << "i = " << i << ", rgb[i] = " << (int) rgb[i] << ", rgb2[i] = " << (int) rgb2[i] << "\n";
//    		int diff = abs(rgb[i] - rgb2[i]);
//    		if (diff > maxerr) {
//    			maxerr = diff;
//    		}
//    		counter++;
//    		if (diff > 1) {
////    			cerr << "i = " << i << ", rgb[i] = " << (int) rgb[i] << ", rgb2[i] = " << (int) rgb2[i] << "\n";
//    			ctr2++;
//    		}
//    	}
//    }
//
//    cout << "\n Number of inequalities: " << counter << "\n";
//    cout << "\n Number of inequalities > 1: " << ctr2 << "\n";
//    cout << "Max inequality: " << maxerr << "\n";
//
//    Mat transformed(rows, cols, CV_8UC3, rgb2);
//
//
//    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
//    imshow( "Display window", transformed );                   // Show our image inside it.
//
//    waitKey(0);                                          // Wait for a keystroke in the window
//    return 0;
//}
