import cv2
import numpy as np

img = cv2.imread('inputNG.png')

min = img.min()
max = img.max()

# stretched_image = (img - np.min(img)) * (255 / (np.max(img) - np.min(img)))
stretched_image = (img - min) * (255 / (max - min))

stretched_image = stretched_image.astype(np.uint8)



cv2.imshow("Original",img)
cv2.imshow("Gamma Corrected ",stretched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()