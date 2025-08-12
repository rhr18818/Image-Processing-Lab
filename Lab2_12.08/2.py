import cv2
import numpy as np

img = cv2.imread('inputTS.png')
## Here demo is already in grayscale but for other we need to do that 
# by ourself
thrshhold_value = 50

# binary_img = np.zeros_like(img)
# binary_img[img>=thrshhold_value] = 255
# binary_img[img<thrshhold_value] = 0

_, binary_img = cv2.threshold(img,thrshhold_value,255,cv2.THRESH_BINARY)

## _, hase been used for Python’s tuple unpacking 
# t’s a way to receive multiple values returned by a function
# but ignore the ones you don’t care about.

# 💡 Fun fact:
# The _ is just a variable name. You could call it garbage, ignore, 
# or temp, but _ is the Python “polite way” of saying “I don’t need this.”


cv2.imshow("Original",img)
cv2.imshow("Threshhold",binary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()