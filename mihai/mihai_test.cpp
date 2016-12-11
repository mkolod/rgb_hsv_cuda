#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <iomanip>

using int8 = int;
using cvInt8 = unsigned char;

//Blatantly plagiarised from here:
//https://github.com/alexkuhl/colorspace-conversion-library/blob/master/colorspace_conversion_library.hpp
void rgb_to_hsv_inner( int r, int g, int b, float& h, float& s, float& v )
{
    float fr = r/255.0, fg = g/255.0, fb = b/255.0 ;
    int imax = std::max( std::max( r, g ), b ) ;
    int imin = std::min( std::min( r, g ), b ) ;
    float fmax = imax/255.0 ;
    float fmin = imin/255.0 ;
    float multiplier = ( imin == imax ) ? 0.0 : 60/( fmax - fmin ) ;
        
    if( r == imax )     // red is dominant
    {
        h = ( multiplier*( fg - fb ) + 360 ) ;
        if( h >= 360 ) h -= 360 ;   // take quick modulus, % doesn't work with floats
    }
    else if( g == imax )// green is dominant
        h = multiplier*( fb - fr ) + 120 ;
    else                // blue is dominant
        h = multiplier*( fr - fg ) + 240 ;  
    if( imax == 0 )
        s = 0 ;
    else
        s = 1 - ( fmin/fmax ) ;
    v = fmax ;
}
 
void hsv_to_rgb_inner( float h, float s, float v, int& r, int& g, int& b )
{
    h /= 60 ;
    int hi = (int)h ;
    float f = h - hi ;
    // all the following *255 are to move from [0,1] to [0,255] domains
    // because rgb are assumed [0,255] integers
    int p = std::round( v*( 1 - s )*255 ) ;
    int q = std::round( v*( 1 - f*s )*255 ) ;
    int t = std::round( v*( 1 - ( 1 - f )*s )*255 ) ;
    int iv = std::round( v*255 ) ;
    switch( hi )
    {
        case 0:
            r = iv ; g = t ; b = p ;
            break ;
        case 1:
            r = q ; g = iv ; b = p ;
            break ;
        case 2:
            r = p ; g = iv ; b = t ;
            break ;
        case 3:
            r = p ; g = q ; b = iv ;
            break ;
        case 4:
            r = t ; g = p ; b = iv ;
            break ;
        case 5:
            r = iv ; g = p ; b = q ;
            break ;
    }
}

float * rgb_to_hsv(int rows, int cols, int8 * rgb) {
    float *hsv = (float *) calloc(rows * cols * 3, sizeof(float));

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int index = ((row * cols) + col) * 3;

            rgb_to_hsv_inner(rgb[index], rgb[index + 1], rgb[index + 2], 
                                hsv[index], hsv[index + 1], hsv[index + 2]);
        }
    }

    return hsv;
}

int8 * hsv_to_rgb(int rows, int cols, float * hsv) {
    int8 *rgb = (int8 *) calloc(rows * cols * 3, sizeof(int8));

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int index = ((row * cols) + col) * 3;

            hsv_to_rgb_inner(hsv[index], hsv[index + 1], hsv[index + 2],
                                    rgb[index], rgb[index + 1], rgb[index + 2]);
        }
    }

    return rgb;
}

void printImage(int rows, int cols, int8 * rgb) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = ((i * cols) + j) * 3;

            for (int c = 0; c < 2; ++c) {
                std::cout << std::setfill(' ') << std::setw(3) << rgb[idx + c];
                std::cout << "|";
            }
            std::cout << std::setfill(' ') << std::setw(3) << rgb[idx + 2] << "\n";
        }
        std::cout << "\n";
    }
}

int8* getData(cvInt8 *cvData, int length) {
    int8 *data = (int8 *) calloc(length, sizeof(int8));

    for (int i = 0; i < length; ++i) {
        data[i] = (int8)cvData[i];
    }

    return data;
}

cvInt8* getCvData(int8 *data, int length) {
    cvInt8 *cvData = (cvInt8 *) calloc(length, sizeof(cvInt8));

    for (int i = 0; i < length; ++i) {
        cvData[i] = (cvInt8)data[i];
    }

    return cvData;
}

void processSpiralImage() {
    std::cout << "Processing spiral.jpg\n";
    cv::Mat bgrImage = cv::imread("spiral.jpg", CV_LOAD_IMAGE_COLOR);

    int rows = bgrImage.rows;
    int cols = bgrImage.cols;
    int channels = bgrImage.channels();
    int total = rows * cols * channels;

    cv::Mat rgbImage(cv::Size(rows, cols), CV_8UC3);

    cv::cvtColor(bgrImage, rgbImage, CV_BGR2RGB);

    std::cout << "rows = " << rows << ", cols = " << cols << ", channels = " << channels << "\n\n";

    int8 *rgbData = getData(rgbImage.data, total);

    float *hsvData = rgb_to_hsv(rows, cols, rgbData);
    int8 *reconstructedRgbData = hsv_to_rgb(rows, cols, hsvData);

    cv::Mat transformed(rows, cols, CV_8UC3, getCvData(reconstructedRgbData, total));

    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", transformed );                   // Show our image inside it.

    cv::waitKey(0);                                          // Wait for a keystroke in the window
}

int main() {
    std::cout << "Processing test.bmp\n";
    cv::Mat bgrImage = cv::imread("test.bmp", CV_LOAD_IMAGE_COLOR);

    int rows = bgrImage.rows;
    int cols = bgrImage.cols;
    int channels = bgrImage.channels();
    int total = rows * cols * channels;

    cv::Mat rgbImage(cv::Size(rows, cols), CV_8UC3);

    cv::cvtColor(bgrImage, rgbImage, CV_BGR2RGB);

    std::cout << "rows = " << rows << ", cols = " << cols << ", channels = " << channels << "\n\n";

    int8 *rgbData = getData(rgbImage.data, total);

    std::cout << "Original:\n";
    printImage(rows, cols, rgbData);

    float *hsvData = rgb_to_hsv(rows, cols, rgbData);
    int8 *reconstructedRgbData = hsv_to_rgb(rows, cols, hsvData);
    
    std::cout << "Reconstructed:\n";
    printImage(rows, cols, reconstructedRgbData);

    processSpiralImage();

    return 0;
}