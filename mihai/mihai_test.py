import cv2
import numpy as np

def rgb_to_hsv(arr):
    
    arr = np.array(arr.astype(np.float32))
    
    # need to scale [0, 255] to [0, 1]
    if np.any(arr > 1.0):
        arr /= 255.0
    
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    
    hsv = np.zeros_like(arr)
    
    # element-wise max and min against the channel dimension (HWC layout) 
    maxes = arr.max(-1)
    mins = arr.min(-1)
    
    # chroma
    c = maxes - mins
        
    # to prevent division by zero
    c_nonzero = c > 0.0
    max_nonzero = maxes > 0.0
    
    # hue calculation
    # red channel
    idx = (maxes == r) & c_nonzero
    hsv[idx, 0] = ((g[idx] - b[idx]) / c[idx]) % 6.0
    
    # green channel
    idx = (maxes == g) & c_nonzero
    hsv[idx, 0] = (b[idx] - r[idx]) / c[idx] + 2.0
    
    # blue channel
    idx = (maxes == b) & c_nonzero
    hsv[idx, 0] = (r[idx] - g[idx]) / c[idx] + 4.0
        
    hsv[..., 0] = hsv[..., 0] * 60 # 6.0

    # saturation
    hsv[max_nonzero, 1] = c[max_nonzero] / maxes[max_nonzero]
    
    # value
    hsv[..., 2] = maxes 
    
    return hsv                  

def hsv_to_rgb(arr):
        
    h = arr[..., 0]
    s = arr[..., 1]
    v = arr[..., 2]
    
    rgb_arr = np.zeros_like(arr)
    
    chroma = v * s
    
    h_prime = h * 6.0
        
    x = chroma * ( 1 - np.abs(h_prime % 2 - 1))  
    m = v - chroma
        
    rgb_arr[..., 0] = chroma * (
        np.logical_and(h_prime >= 0, h_prime < 1) + 
        np.logical_and(h_prime >= 5, h_prime < 6)
    ) + x * (
        np.logical_and(h_prime >= 1, h_prime < 2) +  
        np.logical_and(h_prime >= 4, h_prime < 5)       
    )
    
    rgb_arr[..., 1] = chroma * (
        np.logical_and(h_prime >= 1, h_prime < 3)
    ) + x * (
        np.logical_and(h_prime >= 0, h_prime < 1) +  
        np.logical_and(h_prime >= 3, h_prime < 4)       
    )
    
    rgb_arr[..., 2] = chroma * (
        np.logical_and(h_prime >= 3, h_prime < 5)
    ) + x * (
        np.logical_and(h_prime >= 2, h_prime < 3) +  
        np.logical_and(h_prime >= 5, h_prime < 6)        
    )    
    
    for i in xrange(3):
        rgb_arr[..., i] += m
    
    rgb_arr = np.round(rgb_arr * 255.0).astype(np.uint8)
    rgb_arr = np.minimum(rgb_arr, 255)
    
    return rgb_arr


rgb = cv2.cvtColor(cv2.imread('test.bmp'), cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
# plt.imshow(rgb)

reconstructed_rgb = hsv_to_rgb(rgb_to_hsv(rgb))

reconstructed_rgb_opencv = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#cv2.imwrite("test_python.bmp", reconstructed_rgb)

print("Original:\n")
print(rgb)
print("\n")
print("Reconstructed OpenCV:\n")
print(reconstructed_rgb_opencv)
print("\n")
print("Reconstructed Marek:\n")
print(reconstructed_rgb)