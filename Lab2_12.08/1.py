import cv2
import numpy as np

img = cv2.imread('inputNG.png')

img2 = 255 - img



cv2.imshow("Original",img)
cv2.imshow("Negative",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()