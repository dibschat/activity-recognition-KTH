import numpy as np
import cv2
from utils import window, mag_check, dir_check, create_hist_context

# create Haar-cascade object
body_cascade = cv2.CascadeClassifier('cascadG.xml')

# create background-subtraction object
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
kernel = np.ones((5,5),np.uint8)

def dense_flow(fm):
    # initialize variables
    count = 0
    x = y = w = h = 0
    magnitude_histogram = []
    direction_histogram = []
    magnitude_histogram1 = []
    direction_histogram1 = []
    magnitude_histogram2 = []
    direction_histogram2 = []
    magnitude_histogram3 = []
    direction_histogram3 = []
    magnitude_histogram4 = []
    direction_histogram4 = []  
    magnitude_histogram5 = []
    direction_histogram5 = []
    magnitude_histogram6 = []
    direction_histogram6 = []
    magnitude_histogram7 = []
    direction_histogram7 = []
    magnitude_histogram8 = []
    direction_histogram8 = []
    magnitude_histogram9 = []
    direction_histogram9 = []

    # start reading the video
    cap = cv2.VideoCapture(fm)

    # take the first frame and convert it to gray
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # create the HSV color image
    hsvImg = np.zeros_like(frame)
    hsvImg[..., 1] = 255
    
    # play until the user decides to stop
    frame_no = 0
    while True:
        # save the previous frame data
        previousGray = gray
        # get the next frame
        ret , frame = cap.read()
        
        if ret:
            # background-subtraction
            fgmask = fgbg.apply(frame)

            # median-blur
            seg_mask = cv2.medianBlur(fgmask, 5)

            # dilation
            seg_mask = cv2.dilate(seg_mask, kernel, iterations = 1)

            # for drawing bounding box over the entire body
            body = body_cascade.detectMultiScale(gray, 1.05, 3)
            if(len(body)!=0):
                for (x_t,y_t,w_t,h_t) in body:  
                    x, y, w, h = x_t, y_t, w_t, h_t
            
            # convert the frame to gray scale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # exception-handling
            if((x, y, w, h) == (0 ,0, 0, 0)):
                continue

            # calculate the dense optical flow
            flow = cv2.calcOpticalFlowFarneback(previousGray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
            # obtain the flow magnitude and direction angle
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag = cv2.bitwise_and(mag, mag, mask = seg_mask)
            ang = cv2.bitwise_and(ang, ang, mask = seg_mask)

            # scaling
            ang=((ang*180)/(np.pi/2))%180
            
            # find the intersection points to draw the 3x3 grid
            k=1
            if(w%3==0):
                k=0
            c_x1 = x+(w//3)+k
            c_x2 = (x+2*(w//3))+k
            
            k=1
            if(h%3==0):
                k=0
            c_y1 = y+(h//3)+k
            c_y2 = (y+2*(h//3))+k            

            flag1=flag2=flag3=flag4=0
            if(x-5>=0):
                x-=5
                flag1=1
                if(x+w+10<ang.shape[1]):
                    w+=10
                    flag2=1                    
            if(y-5>=0):
                y-=5
                flag3=1
                if(y+h+10<ang.shape[0]):
                    h+=10
                    flag4=1

            # extract the region-of-interests corresponding to the 3x3 grids
            roi_mag1 = mag[y:c_y1, x:c_x1]
            roi_mag2 = mag[y:c_y1, c_x1:c_x2]
            roi_mag3 = mag[y:c_y1, c_x2:x+w]
            roi_mag4 = mag[c_y1:c_y2, x:c_x1]
            roi_mag5 = mag[c_y1:c_y2, c_x1:c_x2]
            roi_mag6 = mag[c_y1:c_y2, c_x2:x+w]
            roi_mag7 = mag[c_y2:y+h, x:c_x1]
            roi_mag8 = mag[c_y2:y+h, c_x1:c_x2]
            roi_mag9 = mag[c_y2:y+h, c_x2:x+w]
            roi_dir1 = ang[y:c_y1, x:c_x1]
            roi_dir2 = ang[y:c_y1, c_x1:c_x2]
            roi_dir3 = ang[y:c_y1, c_x2:x+w]
            roi_dir4 = ang[c_y1:c_y2, x:c_x1]
            roi_dir5 = ang[c_y1:c_y2, c_x1:c_x2]
            roi_dir6 = ang[c_y1:c_y2, c_x2:x+w]
            roi_dir7 = ang[c_y2:y+h, x:c_x1]
            roi_dir8 = ang[c_y2:y+h, c_x1:c_x2]
            roi_dir9 = ang[c_y2:y+h, c_x2:x+w]

            magnitude = np.array(mag).flatten()
            direction = np.array(ang).flatten()
            magnitude1 = np.array(roi_mag1).flatten()
            direction1 = np.array(roi_dir1).flatten()
            magnitude2 = np.array(roi_mag2).flatten()
            direction2 = np.array(roi_dir2).flatten()
            magnitude3 = np.array(roi_mag3).flatten()
            direction3 = np.array(roi_dir3).flatten()
            magnitude4 = np.array(roi_mag4).flatten()
            direction4 = np.array(roi_dir4).flatten()
            magnitude5 = np.array(roi_mag5).flatten()
            direction5 = np.array(roi_dir5).flatten()
            magnitude6 = np.array(roi_mag6).flatten()
            direction6 = np.array(roi_dir6).flatten()
            magnitude7 = np.array(roi_mag7).flatten()
            direction7 = np.array(roi_dir7).flatten()
            magnitude8 = np.array(roi_mag8).flatten()
            direction8 = np.array(roi_dir8).flatten()
            magnitude9 = np.array(roi_mag9).flatten()
            direction9 = np.array(roi_dir9).flatten()

            # create magnitude and direction optical flow histogram per frame for each grid    
            magnitude_histogram, direction_histogram = create_hist_context(magnitude, direction, magnitude_histogram, direction_histogram)
            magnitude_histogram1, direction_histogram1 = create_hist_context(magnitude1, direction1, magnitude_histogram1, direction_histogram1)
            magnitude_histogram2, direction_histogram2 = create_hist_context(magnitude2, direction2, magnitude_histogram2, direction_histogram2)
            magnitude_histogram3, direction_histogram3 = create_hist_context(magnitude3, direction3, magnitude_histogram3, direction_histogram3)
            magnitude_histogram4, direction_histogram4 = create_hist_context(magnitude4, direction4, magnitude_histogram4, direction_histogram4)
            magnitude_histogram5, direction_histogram5 = create_hist_context(magnitude5, direction5, magnitude_histogram5, direction_histogram5)
            magnitude_histogram6, direction_histogram6 = create_hist_context(magnitude6, direction6, magnitude_histogram6, direction_histogram6)
            magnitude_histogram7, direction_histogram7 = create_hist_context(magnitude7, direction7, magnitude_histogram7, direction_histogram7)
            magnitude_histogram8, direction_histogram8 = create_hist_context(magnitude8, direction8, magnitude_histogram8, direction_histogram8)
            magnitude_histogram9, direction_histogram9 = create_hist_context(magnitude9, direction9, magnitude_histogram9, direction_histogram9)
            
            #---------------------------------------------------------#
            # if you wish to see the optical flow frames uncomment the next 3 paragraphs
            '''
            # update the color image
            hsvImg[..., 0] = 0.5 * ang * 180 / np.pi
            hsvImg[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgbImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
            
            #drawing the bounding box
            cv2.rectangle(rgbImg, (x,y), (c_x1,c_y1), (255,0,0), 2)
            cv2.rectangle(rgbImg, (c_x1,y), (c_x2,c_y1), (0,255,0), 2)
            cv2.rectangle(rgbImg, (c_x2,y), (x+w,c_y1), (0,0,255), 2)
            cv2.rectangle(rgbImg, (x,c_y1), (c_x1,c_y2), (255,255,0), 2)
            cv2.rectangle(rgbImg, (c_x1,c_y1), (c_x2,c_y2), (0,255,255), 2)
            cv2.rectangle(rgbImg, (c_x2,c_y1), (x+w,c_y2), (255,0,255), 2)
            cv2.rectangle(rgbImg, (x,c_y2), (c_x1,y+h), (128,255,0), 2)
            cv2.rectangle(rgbImg, (c_x1,c_y2), (c_x2,y+h), (255,128,0), 2)
            cv2.rectangle(rgbImg, (c_x2,c_y2), (x+w,y+h), (0,255,128), 2)
            
            #Display the resulting frame
            cv2.imshow('dense optical flow', np.hstack((frame, rgbImg)))
            '''
            #---------------------------------------------------------#

            frame_no+=1

            # adjusting the bounding box over the POI to facilitate outward motion of the human body
            if(flag1==1):
                x+=5
                if(flag2==1):
                    w-=10
            if(flag3==1):
                y+=5
                if(flag4==1):
                    h-=10

            k = cv2.waitKey(30) & 0xff        
            if k == 27:
                break
        
        else:
            break

    # check the magnitude and direction histograms to have expected shapes
    magnitude_histogram = mag_check(magnitude_histogram)
    magnitude_histogram1 = mag_check(magnitude_histogram1)
    magnitude_histogram2 = mag_check(magnitude_histogram2)
    magnitude_histogram3 = mag_check(magnitude_histogram3)
    magnitude_histogram4 = mag_check(magnitude_histogram4)
    magnitude_histogram5 = mag_check(magnitude_histogram5)
    magnitude_histogram6 = mag_check(magnitude_histogram6)
    magnitude_histogram7 = mag_check(magnitude_histogram7)
    magnitude_histogram8 = mag_check(magnitude_histogram8)
    magnitude_histogram9 = mag_check(magnitude_histogram9)
    direction_histogram = dir_check(direction_histogram)
    direction_histogram1 = dir_check(direction_histogram1)
    direction_histogram2 = dir_check(direction_histogram2)
    direction_histogram3 = dir_check(direction_histogram3)
    direction_histogram4 = dir_check(direction_histogram4)    
    direction_histogram5 = dir_check(direction_histogram5)
    direction_histogram6 = dir_check(direction_histogram6)
    direction_histogram7 = dir_check(direction_histogram7)
    direction_histogram8 = dir_check(direction_histogram8)
    direction_histogram9 = dir_check(direction_histogram9)

    # apply windowing to extract contextual information
    mag_hist = window(magnitude_histogram)
    dir_hist = window(direction_histogram)
    mag_hist1 = window(magnitude_histogram1)
    dir_hist1 = window(direction_histogram1)
    mag_hist2 = window(magnitude_histogram2)
    dir_hist2 = window(direction_histogram2)
    mag_hist3 = window(magnitude_histogram3)
    dir_hist3 = window(direction_histogram3)
    mag_hist4 = window(magnitude_histogram4)
    dir_hist4 = window(direction_histogram4)
    mag_hist5 = window(magnitude_histogram5)
    dir_hist5 = window(direction_histogram5)
    mag_hist6 = window(magnitude_histogram6)
    dir_hist6 = window(direction_histogram6)
    mag_hist7 = window(magnitude_histogram7)
    dir_hist7 = window(direction_histogram7)
    mag_hist8 = window(magnitude_histogram8)
    dir_hist8 = window(direction_histogram8)
    mag_hist9 = window(magnitude_histogram9)
    dir_hist9 = window(direction_histogram9)

    # calculate the mean of the magnitude and direction histograms for each 3x3 grids
    mag_avg_hist = np.mean(mag_hist, axis=0)
    dir_avg_hist = np.mean(dir_hist, axis=0)
    mag_avg_hist1 = np.mean(mag_hist1, axis=0)
    dir_avg_hist1 = np.mean(dir_hist1, axis=0)
    mag_avg_hist2 = np.mean(mag_hist2, axis=0)
    dir_avg_hist2 = np.mean(dir_hist2, axis=0)
    mag_avg_hist3 = np.mean(mag_hist3, axis=0)
    dir_avg_hist3 = np.mean(dir_hist3, axis=0)
    mag_avg_hist4 = np.mean(mag_hist4, axis=0)
    dir_avg_hist4 = np.mean(dir_hist4, axis=0)
    mag_avg_hist5 = np.mean(mag_hist5, axis=0)
    dir_avg_hist5 = np.mean(dir_hist5, axis=0)
    mag_avg_hist6 = np.mean(mag_hist6, axis=0)
    dir_avg_hist6 = np.mean(dir_hist6, axis=0)
    mag_avg_hist7 = np.mean(mag_hist7, axis=0)
    dir_avg_hist7 = np.mean(dir_hist7, axis=0)
    mag_avg_hist8 = np.mean(mag_hist8, axis=0)
    dir_avg_hist8 = np.mean(dir_hist8, axis=0)
    mag_avg_hist9 = np.mean(mag_hist9, axis=0)
    dir_avg_hist9 = np.mean(dir_hist9, axis=0)

    # calculate the standard deviation of the magnitude and direction histograms for each 3x3 grids
    mag_std_hist = np.std(mag_hist, axis=0)
    dir_std_hist = np.std(dir_hist, axis=0)
    mag_std_hist1 = np.std(mag_hist1, axis=0)
    dir_std_hist1 = np.std(dir_hist1, axis=0)
    mag_std_hist2 = np.std(mag_hist2, axis=0)
    dir_std_hist2 = np.std(dir_hist2, axis=0)
    mag_std_hist3 = np.std(mag_hist3, axis=0)
    dir_std_hist3 = np.std(dir_hist3, axis=0)
    mag_std_hist4 = np.std(mag_hist4, axis=0)
    dir_std_hist4 = np.std(dir_hist4, axis=0)
    mag_std_hist5 = np.std(mag_hist5, axis=0)
    dir_std_hist5 = np.std(dir_hist5, axis=0)
    mag_std_hist6 = np.std(mag_hist6, axis=0)
    dir_std_hist6 = np.std(dir_hist6, axis=0)
    mag_std_hist7 = np.std(mag_hist7, axis=0)
    dir_std_hist7 = np.std(dir_hist7, axis=0)
    mag_std_hist8 = np.std(mag_hist8, axis=0)
    dir_std_hist8 = np.std(dir_hist8, axis=0)
    mag_std_hist9 = np.std(mag_hist9, axis=0)
    dir_std_hist9 = np.std(dir_hist9, axis=0)

    # concatenate all the histogram features to get the contextual descriptor for 3x3 grids
    histogram = mag_avg_hist
    histogram = np.hstack((histogram, mag_std_hist))
    histogram = np.hstack((histogram, dir_avg_hist))
    histogram = np.hstack((histogram, dir_std_hist))
    histogram = np.hstack((histogram, mag_avg_hist1))
    histogram = np.hstack((histogram, mag_std_hist1))
    histogram = np.hstack((histogram, dir_avg_hist1))
    histogram = np.hstack((histogram, dir_std_hist1))
    histogram = np.hstack((histogram, mag_avg_hist2))
    histogram = np.hstack((histogram, mag_std_hist2))
    histogram = np.hstack((histogram, dir_avg_hist2))
    histogram = np.hstack((histogram, dir_std_hist2))
    histogram = np.hstack((histogram, mag_avg_hist3))
    histogram = np.hstack((histogram, mag_std_hist3))
    histogram = np.hstack((histogram, dir_avg_hist3))
    histogram = np.hstack((histogram, dir_std_hist3))
    histogram = np.hstack((histogram, mag_avg_hist4))
    histogram = np.hstack((histogram, mag_std_hist4))
    histogram = np.hstack((histogram, dir_avg_hist4))
    histogram = np.hstack((histogram, dir_std_hist4))
    histogram = np.hstack((histogram, mag_avg_hist5))
    histogram = np.hstack((histogram, mag_std_hist5))
    histogram = np.hstack((histogram, dir_avg_hist5))
    histogram = np.hstack((histogram, dir_std_hist5))
    histogram = np.hstack((histogram, mag_avg_hist6))
    histogram = np.hstack((histogram, mag_std_hist6))
    histogram = np.hstack((histogram, dir_avg_hist6))
    histogram = np.hstack((histogram, dir_std_hist6))
    histogram = np.hstack((histogram, mag_avg_hist7))
    histogram = np.hstack((histogram, mag_std_hist7))
    histogram = np.hstack((histogram, dir_avg_hist7))
    histogram = np.hstack((histogram, dir_std_hist7))
    histogram = np.hstack((histogram, mag_avg_hist8))
    histogram = np.hstack((histogram, mag_std_hist8))
    histogram = np.hstack((histogram, dir_avg_hist8))
    histogram = np.hstack((histogram, dir_std_hist8))
    histogram = np.hstack((histogram, mag_avg_hist9))
    histogram = np.hstack((histogram, mag_std_hist9))
    histogram = np.hstack((histogram, dir_avg_hist9))
    histogram = np.hstack((histogram, dir_std_hist9))
    
    cv2.destroyAllWindows()
    cap.release()
    return histogram, frame_no