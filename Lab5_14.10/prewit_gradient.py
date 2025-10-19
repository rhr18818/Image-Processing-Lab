import cv2
import numpy as np
from scipy.signal import convolve2d

# --- Load Test image ---
orginal_img = cv2.imread("Lena.jpg")


# --- Convert to Grayscale if needed ---
gray_original_img = cv2.cvtColor(orginal_img, cv2.COLOR_BGR2GRAY)


prewitt_x = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]], dtype=np.float32)

prewitt_y = np.array([[1, 1, 1],
                            [0, 0, 0],
                            [-1, -1, -1]], dtype=np.float32)


# --- Apply prewit Convolution Filtering ---

#weighted 3*3
Gx = convolve2d(gray_original_img, prewitt_x, mode='same', boundary='symm')
Gy = convolve2d(gray_original_img, prewitt_y, mode='same', boundary='symm')

#compute gradient magnitude
prewitt_magnitude = np.sqrt(Gx**2+Gy**2)

#clip value to normal range--- if not applied image will be black ---> negative value
Gx = np.clip(np.abs(Gx),0,255).astype(np.uint8)
Gy = np.clip(np.abs(Gy),0,255).astype(np.uint8)
prewit_diplay = np.clip(prewitt_magnitude,0,255).astype(np.uint8)


cv2.imshow("Original Image",gray_original_img)
cv2.imshow("Prewit in X axis", Gx.astype(np.uint8))
cv2.imshow("Prewit in Y axis", Gy.astype(np.uint8))
cv2.imshow("Prewit with Gradient", prewit_diplay.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()