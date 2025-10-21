import cv2
import numpy as np
from scipy.signal import convolve2d

#Load image
orginal_img = cv2.imread("Lena.jpg")
median_img  = cv2.imread("med.jpg")

#convert to grayscale if needed
gray_original_img = cv2.cvtColor(orginal_img,cv2.COLOR_BGR2GRAY)

#weighted avg kernel
weighted_kernel = np.array([[1,2,1],
                            [2,4,2],
                            [1,2,1]],dtype=np.float32)

weighted_kernel /= np.sum(weighted_kernel)

def gaussian_kernel(size,sigma):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.) 
    xx,yy = np.meshgrid(ax,ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2)) 
    
    return kernel/np.sum(kernel)

gauss_kernel = gaussian_kernel(5,1.5)

#apply convolution filter

#in grayscale images
weighted_filtered  = convolve2d(gray_original_img,weighted_kernel,mode='same',boundary='fill',fillvalue=0)

gaussian_filtered   = convolve2d(gray_original_img,gauss_kernel,mode='same',boundary='fill',fillvalue=0)

#color image
b,g,r  = cv2.split(orginal_img)

##hsv
hsv_img = cv2.cvtColor(orginal_img,cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv_img)

b_avg  = convolve2d(b,weighted_kernel,mode='same',boundary='fill',fillvalue=0)
g_avg  = convolve2d(g,weighted_kernel,mode='same',boundary='fill',fillvalue=0)
r_avg  = convolve2d(r,weighted_kernel,mode='same',boundary='fill',fillvalue=0)
bgr_avg  = cv2.merge((b_avg.astype(np.uint8),
                      g_avg.astype(np.uint8),
                      r_avg.astype(np.uint8)))

b_blur = convolve2d(b, gauss_kernel, mode='same', boundary='fill',fillvalue=0)
g_blur = convolve2d(g, gauss_kernel, mode='same', boundary='fill',fillvalue=0)
r_blur = convolve2d(r, gauss_kernel, mode='same', boundary='fill',fillvalue=0)
bgr_blur = cv2.merge((b_blur.astype(np.uint8),
                     g_blur.astype(np.uint8),
                     r_blur.astype(np.uint8)))


#median filter
median_blur  = cv2.medianBlur(median_img,ksize=5)


#gray difference show
# cv2.imshow("Original Image",orginal_img)
# cv2.imshow("Weighted Average Filter", weighted_filtered.astype(np.uint8))
# cv2.imshow("Gaussian Filter", gaussian_filtered.astype(np.uint8))
# cv2.imshow("Median Filter", median_img)


#color image difference show
cv2.imshow("Original Image",orginal_img)
cv2.imshow("Weighted Average Filter",bgr_avg)
cv2.imshow("Gaussian Filter",bgr_blur)



cv2.waitKey(0)
cv2.destroyAllWindows()