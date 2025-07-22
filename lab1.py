import numpy as np
import cv2

image = cv2.imread('Lena.jpeg')

image2 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2.imwrite('Grey_Lena.jpg',image2)
print(image)

cv2.imshow('Test',image2)
cv2.waitKey(0)
cv2.destroyAllWindows()