import numpy as np
import cv2
import matplotlib.pyplot as plt

def calculate_histogram(image):
    histogram,_ = np.histogram(image,bins=256,range=(0,256))
    return histogram


def plot_histogram_rgb(count_r,count_v,count_eq_r,count_eq_v):
    hists = [count_r,count_v,count_eq_r,count_eq_v]
    colors = ['b','g','b','g']
    titles = ['Red Original Histogram','Val Original Histogram','Red Eqalized Histogram','Val Eqalized Histogram']
    
    plt.figure(figsize=(16,8))
    
    for i in range(4):
        plt.subplot(2, 2, i+1) # row, column , iteration 
        plt.bar(range(256),hists[i].ravel(),color=colors[i],width=1)
        plt.title(titles[i])
        plt.xlabel("Pixel values")
        plt.ylabel("Frequency")
        plt.xlim([0, 256]) #not necessary range fixed in x axis
        
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()

if __name__ == "__main__":
    img_path = 'Histo.png'
    
    img_bgr = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
    
    #opencv  b,g,r format read
    b,g,r = cv2.split(img_bgr)
    h,s,v = cv2.split(img_hsv)
    
    #for plotting
    hist_r = calculate_histogram(r)
    hist_v = calculate_histogram(v)
    hist_b = calculate_histogram(b)
    

    r_eq = cv2.equalizeHist(r)
    v_eq = cv2.equalizeHist(v)
    
    hist_r_eq = calculate_histogram(r_eq)
    hist_v_eq = calculate_histogram(v_eq)
    
    
    plot_histogram_rgb(hist_r,hist_b,hist_r_eq,hist_v_eq)
    
    
    img_bgr_eq =cv2.merge([b,g,r_eq])
    
    img_hsv_eq =cv2.merge([h,s,v_eq])
    img_hsv_eq_bgr= cv2.cvtColor(img_hsv_eq,cv2.COLOR_HSV2BGR)
    
    
    cv2.imshow("input RGB image",img_bgr)
    cv2.imshow("RGB",img_bgr_eq)
    cv2.imshow("HSV",img_hsv_eq_bgr)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    