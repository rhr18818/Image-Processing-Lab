import cv2
import numpy as np

img = cv2.imread('inputLG.jpg')
float_img = img.astype(np.float32)
c = 255/np.log(1+np.max(float_img))

#print(c)
log_img = c *np.log(1+float_img) # s
normalize_grey_img = np.clip(log_img,0,255) #normalize the ragnge can also use - np.normalize/cv2.normalize
output_img = normalize_grey_img.astype(np.uint8)

cv2.imwrite('Logged_inputLG.jpg',output_img)

cv2.imshow("Original",img)
cv2.imshow("Log Output Image",output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()