import cv2
import numpy as np
from scipy.signal import convolve2d

img = cv2.imread("moon.jpg")
original_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

print(original_img.shape)

kernel1 = np.array([[1,2,1],
                    [2,4,2],
                    [1,2,1]],dtype=np.float32)
kernel1 /= np.sum(kernel1)

kernel2 = np.array([[1,1,1],
                    [1,-8,1],
                    [1,1,1]],dtype=np.float32)


i1 = convolve2d(original_img,kernel1,mode='same',boundary='symm')

image1 = np.clip(i1,0,255).astype(np.uint8)

i2 = convolve2d(image1,kernel2,mode='same',boundary='symm')

image2 = np.clip(np.abs(i2),0,255).astype(np.uint8)



# cv2.imshow("Origianl Image",original_img)
# cv2.imshow("Image 1 after filter 1",image1.astype(np.uint8))
# cv2.imshow("Image 2 after filter 2",image2)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

