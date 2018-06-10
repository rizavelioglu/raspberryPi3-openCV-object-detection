from collections import deque
import numpy as np
import argparse
import cv2
import time
from picamera.array import PiRGBArray
from picamera import PiCamera
#raspberry pi configuration 
camera = PiCamera()
#camera.resolution = (640, 480)
camera.resolution = (320, 240)
camera.framerate = 32
#rawCapture = PiRGBArray(camera, size=(640, 480))
rawCapture = PiRGBArray(camera, size=(320, 240)) 
# allow the camera to warmup
time.sleep(0.1)

#HSV values, ([0-179],[0-255],[255])
lower = {'red':(166, 180, 120), 'green':(33, 170, 102), 'blue':(85, 120, 120)}
upper = {'red':(186,255,255), 'green':(75,255,255), 'blue':(135,255,255)}
colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0)}

#capturing frames
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the current frame
    frame = frame.array
 
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    #for each color in dictionary check object in frame
    for key, value in upper.items():
        # construct a mask for the color from dictionary`1, then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        kernel = np.ones((9,9),np.uint8)
        mask = cv2.inRange(hsv, lower[key], upper[key])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
               
        # find contours in the mask and initialize the current
        # (x, y) center of the object
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
       
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
       
            # only proceed if the width and height meets a minimum size. Correct this value for your obect's size
            if w and h > 5:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.drawContours(frame, [box], 0, colors[key], 2)
                cv2.circle(frame, center, 5, colors[key], -1)
                cv2.putText(frame,key + " object", (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
 
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    cv2.imshow("mask", mask)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
