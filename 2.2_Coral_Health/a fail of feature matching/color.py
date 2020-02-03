import cv2
import numpy as np
image = cv2.imread('coral1.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_range = np.array([0,100,100])
upper_range = np.array([200,255,255])
mask = cv2.inRange(hsv, lower_range, upper_range)
cv2.imshow('image_window_name', image)
cv2.imshow('mask_window_name', mask)
cv2.waitKey(0)
