import os
import numpy as np 
import cv2
import scipy
import math
import matplotlib.pyplot as plt
from utils import window, Hog_descriptor

# Orientation based features (A)
def HOG_central(fm):   
    cap = cv2.VideoCapture(fm)
    hist = []
    while True:
        ret,frame1 = cap.read()

        if frame1 is None:
            break

        image = frame1
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 128), interpolation=cv2.INTER_CUBIC)

        hog = Hog_descriptor(img, cell_size=8, bin_size=9)
        vector, image = hog.extract()
        vector = np.sum(np.array(vector), axis=0)
        hist.append(vector)

        # uncomment the next 2 lines to see the HOG images per frame
        #plt.imshow(image, cmap=plt.cm.gray)
        #plt.show()

        if cv2.waitKey(30)==27 & 0xff:
            break

    # calculate mean and average of HOG magnitude and direction histograms
    hist = np.array(hist)
    hog_avg_hist = np.mean(hist, axis=0)
    hog_std_hist = np.std(hist, axis=0)

    histogram = hog_avg_hist
    histogram = np.hstack((histogram, hog_std_hist))

    cv2.destroyAllWindows()
    cap.release()
    return histogram 


# orientation based contextual features (C)
def HOG_context(fm):   
    cap = cv2.VideoCapture(fm)
    hist = []
    count = 0
    while True:
        ret,frame1 = cap.read()

        if frame1 is None:
            break

        image = frame1
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 128), interpolation=cv2.INTER_CUBIC)

        hog = Hog_descriptor(img, cell_size=8, bin_size=9)
        vector, image = hog.extract()
        vector = np.sum(np.array(vector), axis=0)
        hist.append(vector)
        
        # uncomment the next 2 lines to see the HOG images per frame
        #plt.imshow(image, cmap=plt.cm.gray)
        #plt.show()

        if cv2.waitKey(30)==27 & 0xff:
            break

    hist = np.array(hist)

    # calcuate contextual information
    hist = window(hist)

    # calculate mean and average of HOG magnitude and direction histograms
    hog_avg_hist = np.mean(hist, axis=0)
    hog_std_hist = np.std(hist, axis=0)

    histogram = hog_avg_hist
    histogram = np.hstack((histogram, hog_std_hist))

    cv2.destroyAllWindows()
    cap.release()
    return histogram
    #------------------------------------------------------------------------------------------------