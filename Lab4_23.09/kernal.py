import cv2
import numpy as np
from scipy.signal import convolve2d

# --- Load Test image ---
orginal_img = cv2.imread("Lena.jpg")
median_img = cv2.imread("med.png")

# --- Convert to Grayscale if needed ---
gray_original_img = cv2.cvtColor(orginal_img, cv2.COLOR_BGR2GRAY)
gray_median_img = cv2.cvtColor(median_img, cv2.COLOR_BGR2GRAY)


# --- Weighted Average Filter (example 3x3) ---
weighted_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]], dtype=np.float32)
weighted_kernel /= np.sum(weighted_kernel)  # normalize

# --- Weighted Average Filter (example 5x5) ---
weighted_kernel_5 = np.array([[1,  4,  6,  4, 1],
                                [4, 16, 24, 16, 4],
                                [6, 24, 36, 24, 6],
                                [4, 16, 24, 16, 4],
                                [1,  4,  6,  4, 1]], dtype=np.float32)
weighted_kernel_5 /= np.sum(weighted_kernel_5) # normalize


# --- Gaussian Filter ---
def gaussian_kernel(size, sigma):
    """Generates a Gaussian kernel."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.) #can also use range instead of arange !! // --> floating division flor
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2)) # exp means --> power e^ u2+v2/2s2
    return kernel / np.sum(kernel)

gauss_kernel = gaussian_kernel(size=5, sigma=1.5)

gauss_kernel_7 = gaussian_kernel(size=7, sigma=1.5)

# print("-------------------")
# print("Gaussian Kernel:\n", gauss_kernel)
# print("Size of Gaussian Kernel:", gauss_kernel.shape)
# print("-------------------")




# --- Apply Convolution Filtering ---

#Color image apply
b,g,r = cv2.split(orginal_img)


#weighted 3*3
b_avg = convolve2d(b, weighted_kernel, mode='same', boundary='fill',fillvalue=0)
g_avg = convolve2d(g, weighted_kernel, mode='same', boundary='fill',fillvalue=0)
r_avg = convolve2d(r, weighted_kernel, mode='same', boundary='fill',fillvalue=0)
bgr_avg_3 = cv2.merge((b_avg.astype(np.uint8),
                     g_avg.astype(np.uint8),
                     r_avg.astype(np.uint8)))

#weighted 5*5
b_avg_5 = convolve2d(b, weighted_kernel_5, mode='same', boundary='fill',fillvalue=0)
g_avg_5 = convolve2d(g, weighted_kernel_5, mode='same', boundary='fill',fillvalue=0)
r_avg_5 = convolve2d(r, weighted_kernel_5, mode='same', boundary='fill',fillvalue=0)
bgr_avg_5 = cv2.merge((b_avg_5.astype(np.uint8),
                     g_avg_5.astype(np.uint8),
                     r_avg_5.astype(np.uint8)))




#gaussian 5*5
b_blur = convolve2d(b, gauss_kernel, mode='same', boundary='fill',fillvalue=0)
g_blur = convolve2d(g, gauss_kernel, mode='same', boundary='fill',fillvalue=0)
r_blur = convolve2d(r, gauss_kernel, mode='same', boundary='fill',fillvalue=0)
bgr_blur_5 = cv2.merge((b_blur.astype(np.uint8),
                     g_blur.astype(np.uint8),
                     r_blur.astype(np.uint8)))


#gassian 7*7
b_blur_7 = convolve2d(b, gauss_kernel_7, mode='same', boundary='fill',fillvalue=0)
g_blur_7 = convolve2d(g, gauss_kernel_7, mode='same', boundary='fill',fillvalue=0)
r_blur_7 = convolve2d(r, gauss_kernel_7, mode='same', boundary='fill',fillvalue=0)
bgr_blur_7 = cv2.merge((b_blur_7.astype(np.uint8),
                     g_blur_7.astype(np.uint8),
                     r_blur_7.astype(np.uint8)))



# #kernal size difference show -- wighted
# cv2.imshow("Original Image",orginal_img)
# cv2.imshow("Weighted Average Filter 3*3", bgr_avg_3.astype(np.uint8))
# cv2.imshow("Weighted Average Filter 5*5", bgr_avg_5.astype(np.uint8))


# kernal size difference show -- gaussian
cv2.imshow("Original Image",orginal_img)
cv2.imshow("Gaussian Filter 5*5", bgr_blur_5.astype(np.uint8))
cv2.imshow("Gaussian Filter 7*7", bgr_blur_7.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()