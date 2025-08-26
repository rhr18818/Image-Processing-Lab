import cv2
import numpy as np

img = cv2.imread('inputNG.png')

gamma = 0.5
gamma_corrected_image = np.power(img / 255.0, gamma) * 255.0
gamma_corrected_image = gamma_corrected_image.astype(np.uint8)



cv2.imshow("Original",img)
cv2.imshow("Gamma Corrected ",gamma_corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()