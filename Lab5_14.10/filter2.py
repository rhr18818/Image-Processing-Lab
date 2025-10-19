import cv2
import numpy as np
from scipy.signal import convolve2d

# --- Load Test image ---
orginal_img = cv2.imread("Lena.jpg")


# --- Convert to Grayscale if needed ---
gray_original_img = cv2.cvtColor(orginal_img, cv2.COLOR_BGR2GRAY)



# --- Laplacian Filter (example 3x3) ---
# laplacian_kernel = np.array([[0, 1, 0],
#                             [1, -4, 1],
#                             [0, 1, 0]], dtype=np.float32)

## show the edges only!!!

laplacian_kernel = np.array([[0, 1, 0],
                            [1, -5, 1],
                            [0, 1, 0]], dtype=np.float32)

### show the original image with sharped version


# --- Apply Convolution Filtering ---

#weighted 3*3
laplacian_img = convolve2d(gray_original_img, laplacian_kernel, mode='same', boundary='symm')

#clip value to normal range--- if not applied image will be black ---> negative value
laplacian_img = np.clip(np.abs(laplacian_img),0,255).astype(np.uint8)


cv2.imshow("Original Image",gray_original_img)
cv2.imshow("Laplacian Filter 3*3", laplacian_img.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()