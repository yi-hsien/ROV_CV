import cv2
import numpy as np

def nothing(x):
    pass

image = cv2.imread('coral1.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

cv2.namedWindow('image')
cv2.resizeWindow('image', 600, 600)
cv2.createTrackbar('H_low','image', 0, 255, nothing)
cv2.createTrackbar('S_low','image', 0, 255, nothing)
cv2.createTrackbar('V_low','image', 0, 255, nothing)
cv2.createTrackbar('H_high','image', 0, 255, nothing)
cv2.createTrackbar('S_high','image', 0, 255, nothing)
cv2.createTrackbar('V_high','image', 0, 255, nothing)

cv2.setTrackbarPos('H_high','image',183)
cv2.setTrackbarPos('S_high','image',255)
cv2.setTrackbarPos('V_high','image',255)
cv2.imshow('image_window_name', image)
while(1):
    lower_range = np.array([cv2.getTrackbarPos('H_low', 'image'),cv2.getTrackbarPos('S_low', 'image'),cv2.getTrackbarPos('V_low', 'image')])
    upper_range = np.array([cv2.getTrackbarPos('H_high', 'image'),cv2.getTrackbarPos('S_high', 'image'),cv2.getTrackbarPos('V_high', 'image')])
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    cv2.imshow('mask_window_name', mask)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    #cv2.waitKey(0)
    #lower_range = np.array([0,100,100])
    #upper_range = np.array([200,255,255])

cv2.destroyAllWindows()






