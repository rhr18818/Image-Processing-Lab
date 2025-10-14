import numpy as np
import cv2
import matplotlib.pyplot as plt


def calculate_histogram(image):

    # Calculate the histogram-- count the intensity
    histogram, bins = np.histogram(image, bins=256, range=(0, 256))
    
    return histogram

def plot_histogram_rgb(hist_b, hist_g, hist_r):
    """
    Plots 3 subplots for pre-computed histograms of B, G, R channels.
    """
    hists = [hist_b, hist_g, hist_r]
    colors = ['b', 'g', 'r']
    titles = ['Blue Channel Histogram', 'Green Channel Histogram', 'Red Channel Histogram']

    plt.figure(figsize=(15, 5))
    
    for i in range(3):
        plt.subplot(1, 3, i+1) # row, column , iteration 
        plt.bar(range(256), hists[i].ravel(), color=colors[i], width=1) # ravel for safety reason 2D to 1D ,but here no use , our is 1D
        plt.title(titles[i])
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.xlim([0, 256])#not necessary range fixed in x axis
    
    plt.tight_layout()
    # plt.show()
    plt.show(block=False)   # Show window but don’t block execution
    plt.pause(2)            # Keep it open for 2 seconds
    plt.close()
    # plt.savefig("hist_output.png")

if __name__ == "__main__":
    img_path = 'Picture1.png'
    img_bgr = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
    
    #open-cv always read iamge as b,g,r format
    b,g,r = cv2.split(img_bgr)
    
    # h,s,v = cv2.split(img_hsv)
    # v_eq = cv2.equalizeHist(v)
    # img_hsv_eq =cv2.merge([h,s,v_eq])
    # img_hsv_eq_bgr= cv2.cvtColor(img_hsv_eq,cv2.COLOR_HSV2BGR)
    
    hist_b = calculate_histogram(b)
    hist_g = calculate_histogram(g)
    hist_r = calculate_histogram(r)

    plot_histogram_rgb(hist_b,hist_g,hist_r)
    
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)
    
    hist_b_eq = calculate_histogram(b_eq)
    hist_g_eq = calculate_histogram(g_eq)
    hist_r_eq = calculate_histogram(r_eq)
    
    plot_histogram_rgb(hist_b_eq,hist_g_eq,hist_r_eq)
    
    img_bgr_eq =cv2.merge([b_eq,g_eq,r_eq])
    
    cv2.imshow("input RGB image",img_bgr)
    cv2.imshow("Changed Image",img_bgr_eq)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
