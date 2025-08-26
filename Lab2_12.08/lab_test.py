import cv2 
import numpy as np

img = cv2.imread('inputLG.jpg')

float_img = img.astype(np.float32)
gamma = 2.2
gamma_img= np.power(float_img/255.0,gamma)*255.0


gamma_img= gamma_img.astype(np.uint8)

cv2.imshow("Original",img)
cv2.imshow("Gamma Corrected ",gamma_img)
cv2.waitKey(0)
cv2.destroyAllWindows()