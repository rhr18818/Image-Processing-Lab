import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_histogram(image):
    histogram,_ = np.histogram(image,bins=256,range=(0,256))
    return histogram


def plot_histogram_rgb(count_b,count_g,count_r):
    hists = [count_b,count_g,count_r]
    colors = ['b','g','r']
    titles = ['Blue Channel Histogram','Green Channel Histogram','Red Channel Histogram']
    
    plt.figure(figsize=(15,5))
    
    for i in range(3):
        plt.subplot(1, 3, i+1) # row, column , iteration 
        plt.bar(range(256),hists[i].ravel(),color=colors[i],width=1) # ravel for safety reason 2D to 1D ,but here no use , our is 1D
        plt.title(titles[i])
        plt.xlabel("Pixel values")
        plt.ylabel("Frequency")
        plt.xlim([0, 256]) #not necessary range fixed in x axis
        
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()

if __name__ == "__main__":
    img_path = '/home/rhr18818/Desktop/All/Image Processing/Lab3_26.08 /Picture1.png'
    
    img_bgr = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
    
    #opencv  b,g,r format read
    b,g,r = cv2.split(img_bgr)
    
    #for plotting
    hist_b = calculate_histogram(b)
    hist_g = calculate_histogram(g)
    hist_r = calculate_histogram(r)
    
    plot_histogram_rgb(hist_b,hist_g,hist_r)
    
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    
    img_bgr_eq = cv2.merge([b_eq,g_eq,r_eq])
    
    cv2.imshow("Input RGB Image",img_bgr)
    cv2.imshow("Chnaged Image",img_bgr_eq)
    cv2.waitKey(0)


    