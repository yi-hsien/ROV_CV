import cv2
import numpy as np

def nothing(x):
    pass

kernel_size = 5
cap = cv2.VideoCapture(0)
fr,frame = cap.read()
line_edges = np.copy(frame)*0


cv2.namedWindow('image')
cv2.createTrackbar('thresh_low','image', 0, 255, nothing)
cv2.createTrackbar('thresh_high','image', 0, 255, nothing)

cv2.setTrackbarPos('thresh_low','image',20)
cv2.setTrackbarPos('thresh_high','image',150)
while(True):
    fr,frame = cap.read()
    grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(grey_image, (kernel_size,kernel_size), 0)

    edges = cv2.Canny(blur_gray, cv2.getTrackbarPos('thresh_low', 'image'), cv2.getTrackbarPos('thresh_high', 'image'))
    #low_thresh = 20
    #high_thresh = 150

    #edges = cv2.Canny(blur_gray, low_thresh, high_thresh)

    imshape = frame.shape
    vertices = np.array(
        [
            [
                (0, 0),
                (0, imshape[0]),
                (imshape[1], 0),
                (imshape[1], imshape[0]),
            ]
        ], dtype=np.int32
    )

    # blank image to do operations on
    blank = np.zeros(edges.shape, np.uint8)

    # region of interest as a trapezoid
    edges2 = cv2.fillPoly(blank, vertices, 255)

    # anding the ROI with edges
    masked_edges = cv2.bitwise_and(edges, edges2)

    rho = 1
    theta = np.pi/180
    line_image = np.copy(frame)*0
    count = 0
    lines = cv2.HoughLines(masked_edges, rho,theta,25)
    if lines is not None:
        for line in lines:
            if(count > 8):
                break
            r,t = line[0]
            a = np.cos(t)
            b = np.sin(t)
            x0 = a*r
            y0 = b*r
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(line_image,(x1,y1),(x2,y2),(0,255,255),2)
            line_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
            count = count + 1

    cv2.imshow("ROI", masked_edges)
    cv2.imshow("overlay", line_edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()