import cv2
import numpy as np

def nothing(x):
    pass
# Playing video from file:
# cap = cv2.VideoCapture('vtest.avi')
# Capturing video from webcam:
cap = cv2.VideoCapture(0)
cv2.namedWindow('image')
cv2.createTrackbar('H_low','image', 0, 255, nothing)
cv2.createTrackbar('S_low','image', 0, 255, nothing)
cv2.createTrackbar('V_low','image', 0, 255, nothing)
cv2.createTrackbar('H_high','image', 0, 255, nothing)
cv2.createTrackbar('S_high','image', 0, 255, nothing)
cv2.createTrackbar('V_high','image', 0, 255, nothing)

cv2.setTrackbarPos('H_low','image',90)
cv2.setTrackbarPos('S_low','image',14)
cv2.setTrackbarPos('V_low','image',0)

cv2.setTrackbarPos('H_high','image',183)
cv2.setTrackbarPos('S_high','image',255)
cv2.setTrackbarPos('V_high','image',255)
while(True):
    lower_range = np.array([cv2.getTrackbarPos('H_low', 'image'),cv2.getTrackbarPos('S_low', 'image'),cv2.getTrackbarPos('V_low', 'image')])
    upper_range = np.array([cv2.getTrackbarPos('H_high', 'image'),cv2.getTrackbarPos('S_high', 'image'),cv2.getTrackbarPos('V_high', 'image')])
    # Capture frame-by-frame
    frm, frame = cap.read()


    # Handles the mirroring of the current frame
    frame = cv2.flip(frame,1)

    # Our operations on the frame come here
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    #dilation = cv2.dilate(mask,kernel,iterations = 1)
    # Saves image of the current frame in jpg file
    # name = 'frame' + str(currentFrame) + '.jpg'
    # cv2.imwrite(name, frame)

    # Display the resulting frame
    cv2.imshow('image_window_name', frame)
    cv2.imshow('eroded', erosion)
    cv2.imshow('dilated', mask)
    cv2.imshow('original', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# 90, 14, 0 for blue