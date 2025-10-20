import cv2
import numpy as np
from scipy.signal import convolve2d

orginal_img = cv2.imread("Lena2.jpeg")
# --- Convert to Grayscale if needed ---
gray_original_img = cv2.cvtColor(orginal_img, cv2.COLOR_BGR2GRAY)

laplacian_kernel = np.array([[0, 1, 0],
                            [1, -5, 1],
                            [0, 1, 0]], dtype=np.float32)

laplacian_img = convolve2d(gray_original_img,laplacian_kernel,mode='same',boundary='symm')

laplacian_img = np.clip(np.abs(laplacian_img),0,255).astype(np.uint8)

#color img
b,g,r = cv2.split(orginal_img)
b_l = convolve2d(b,laplacian_kernel,mode='same',boundary='symm')
b_l = np.clip(np.abs(b_l),0,255)

g_l = convolve2d(g,laplacian_kernel,mode='same',boundary='symm')
g_l = np.clip(np.abs(g_l),0,255)

r_l = convolve2d(r,laplacian_kernel,mode='same',boundary='symm')
r_l = np.clip(np.abs(r_l),0,255)

bgr_laplace = cv2.merge((b_l.astype(np.uint8),
                     g_l.astype(np.uint8),
                     r_l.astype(np.uint8)))



cv2.imshow("Original Image",gray_original_img)

cv2.imshow("Laplacian Filter 3*3", laplacian_img)
cv2.imshow("Laplacian Filter in BGR", bgr_laplace)
cv2.waitKey(0)
cv2.destroyAllWindows()