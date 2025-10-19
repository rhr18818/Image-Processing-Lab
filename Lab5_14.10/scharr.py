import cv2
import numpy as np
from scipy.signal import convolve2d

# --- Load Test image ---
orginal_img = cv2.imread("Lena.jpg")


# --- Convert to Grayscale if needed ---
gray_original_img = cv2.cvtColor(orginal_img, cv2.COLOR_BGR2GRAY)


scharr_x = np.array([[-3, 0, 3],
                            [-10, 0, 10],
                            [-3, 0, 3]], dtype=np.float32)

scharr_y = np.array([[10, 10, 10],
                            [0, 0, 0],
                            [-3, -10, -3]], dtype=np.float32)


# --- Apply prewit Convolution Filtering ---

#weighted 3*3
Gx = convolve2d(gray_original_img, scharr_x, mode='same', boundary='symm')
Gy = convolve2d(gray_original_img, scharr_y, mode='same', boundary='symm')

#clip value to normal range--- if not applied image will be black ---> negative value
Gx = np.clip(np.abs(Gx),0,255).astype(np.uint8)
Gy = np.clip(np.abs(Gy),0,255).astype(np.uint8)


cv2.imshow("Original Image",gray_original_img)
cv2.imshow("Scharr in X axis", Gx.astype(np.uint8))
cv2.imshow("Scharr in Y axis", Gy.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()