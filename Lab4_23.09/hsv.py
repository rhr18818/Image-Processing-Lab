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

# --- Gaussian Filter ---
def gaussian_kernel(size, sigma):
    """Generates a Gaussian kernel."""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.) #can also use range instead of arange !! // --> floating division flor
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2)) # exp means --> power e^ u2+v2/2s2
    return kernel / np.sum(kernel)

gauss_kernel = gaussian_kernel(size=5, sigma=1.5)



#Color image apply
b,g,r = cv2.split(orginal_img)

##hsv
hsv_img = cv2.cvtColor(orginal_img,cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv_img)

#weighted--bgr
b_avg = convolve2d(b, weighted_kernel, mode='same', boundary='fill',fillvalue=0)
g_avg = convolve2d(g, weighted_kernel, mode='same', boundary='fill',fillvalue=0)
r_avg = convolve2d(r, weighted_kernel, mode='same', boundary='fill',fillvalue=0)
bgr_avg = cv2.merge((b_avg.astype(np.uint8),
                     g_avg.astype(np.uint8),
                     r_avg.astype(np.uint8)))

#weighted--hsv
s_avg = convolve2d(s, weighted_kernel, mode='same', boundary='fill',fillvalue=0)
v_avg = convolve2d(v, weighted_kernel, mode='same', boundary='fill',fillvalue=0)
hsv_avg = cv2.merge((h.astype(np.uint8),
                     s_avg.astype(np.uint8),
                     v_avg.astype(np.uint8)))
hsv_avg_bgr = cv2.cvtColor(hsv_avg,cv2.COLOR_HSV2BGR)


#gaussian --bgr
b_blur = convolve2d(b, gauss_kernel, mode='same', boundary='fill',fillvalue=0)
g_blur = convolve2d(g, gauss_kernel, mode='same', boundary='fill',fillvalue=0)
r_blur = convolve2d(r, gauss_kernel, mode='same', boundary='fill',fillvalue=0)
bgr_blur = cv2.merge((b_blur.astype(np.uint8),
                     g_blur.astype(np.uint8),
                     r_blur.astype(np.uint8)))

#gaussian --hsv
s_blur = convolve2d(s, gauss_kernel, mode='same', boundary='fill',fillvalue=0)
v_blur = convolve2d(v, gauss_kernel, mode='same', boundary='fill',fillvalue=0)
hsv_blur = cv2.merge((h.astype(np.uint8),
                     s_blur.astype(np.uint8),
                     v_blur.astype(np.uint8)))
hsv_blur_bgr = cv2.cvtColor(hsv_blur,cv2.COLOR_HSV2BGR)

# #color difference show --bgr,hsv Average
# cv2.imshow("Original Image",orginal_img)
# cv2.imshow("Weighted Average Filter-BGR", bgr_avg.astype(np.uint8))
# cv2.imshow("Weighted Average Filter-HSV", hsv_avg_bgr.astype(np.uint8))

#difference calculation 
diff_bgr_hsv = cv2.absdiff(bgr_blur,hsv_blur_bgr)
normalize_diff = cv2.normalize(diff_bgr_hsv,None,0,255,cv2.NORM_MINMAX)
cv2.imshow("Difference btween HSV & BGR",normalize_diff)

#color difference show --bgr,hsv Gaussian
cv2.imshow("Original Image",orginal_img)
cv2.imshow("Gaussian Filter-BGR", bgr_blur.astype(np.uint8))
cv2.imshow("Gaussian Average Filter-HSV", hsv_blur_bgr.astype(np.uint8))


cv2.waitKey(0)
cv2.destroyAllWindows()