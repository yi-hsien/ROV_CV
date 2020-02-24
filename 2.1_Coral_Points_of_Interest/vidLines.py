import cv2
import numpy as np
import math

# function called when trackbar adjusted
def nothing(x):
    pass

# kernel used for HoughLines
kernel_size = 5

# object to open up camera
cap = cv2.VideoCapture(0)

# store the current frame as an image
fr,frame = cap.read()

# empty image object with dim of frame
line_edges = np.copy(frame)*0

# create trackbars for hough line thresholds
cv2.namedWindow('image')
cv2.createTrackbar('thresh_low','image', 0, 255, nothing)
cv2.createTrackbar('thresh_high','image', 0, 255, nothing)

cv2.setTrackbarPos('thresh_low','image',0)
cv2.setTrackbarPos('thresh_high','image',50)


cv2.namedWindow('height')

cv2.createTrackbar('height_low','height', 0, 400, nothing)
cv2.setTrackbarPos('height_low','height',150)

cv2.createTrackbar('height_high','height', 0, 400, nothing)
cv2.setTrackbarPos('height_high','height',0)

# main loop for reading video frames
while(True):
    # first read a frame
    fr,frame = cap.read()
    # convert to gray scale and blur for better performance (less computation)
    grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(grey_image, (kernel_size,kernel_size), 0)

    # extract the edges (areas of high differential contrast) using the trackbar thresholds
    edges = cv2.Canny(blur_gray, cv2.getTrackbarPos('thresh_low', 'image'), cv2.getTrackbarPos('thresh_high', 'image'))

    # get the image dimensions
    imshape = frame.shape

    # make the ROI the entire frame
    vertices = np.array(
        [
            [
                (0, 0),
                (0, imshape[0]),
                (imshape[1], imshape[0]),
                (imshape[1], 0)
            ]
        ], dtype=np.int32
    )

    # blank image to do operations on
    blank = np.zeros(edges.shape, np.uint8)

    # fill the region of interest 
    edges2 = cv2.fillPoly(blank, vertices, 255)

    # anding the ROI with edges to get only the lines in desired region
    masked_edges = cv2.bitwise_and(edges, edges2)

    # lists used for filtering lines
    lineCords = []
    validHLC = []
    validVLC = []
    leftLines = []
    rightLines = []


    rho = 1
    theta = np.pi/180

    line_image = np.copy(frame)*0
    count = 0
    
    # extract lines from edges in ROI
    lines = cv2.HoughLines(masked_edges, rho,theta,25)

    # only try to extract line data if there are lines
    if lines is not None:
        # get the first 40 lines and convert into cartesian coords
        for line in lines:
            if(count > 40):
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
            lineCords.append([(x1,y1), (x2,y2)])
            count = count + 1

    # filter out the valid vertical and horizontal lines
    # if both x coords are within screen width then it is vertical
    # if both y coords are within screen height then it is horizontal
    for line in lineCords:
        if line[0][0] > 0 and line[0][0] < frame.shape[1] and line[1][0] > 0 and line[1][0] < frame.shape[1]:
            validVLC.append(line)
        if line[0][1] > 0 and line[0][1] < frame.shape[0] and line[1][1] > 0 and line[1][1] < frame.shape[0] and len(validHLC) < 4:
            validHLC.append(line)
            
    # filter out "left" and "right" Vertical lines based on center of image, left = blue, right = red, draw the lines as well
    for line in validVLC:
        if line[0][0] < frame.shape[1] / 2 and line[1][0] < frame.shape[1] / 2 and len(leftLines) < 4:
            leftLines.append(line)
            # cv2.line(line_image,(line[0][0],line[0][1]),(line[1][0],line[1][1]),(255,0,0),2)
            # line_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
        elif line[0][0] > frame.shape[1] / 2 and line[1][0] > frame.shape[1] / 2 and len(rightLines) < 4:
            rightLines.append(line)
            # cv2.line(line_image,(line[0][0],line[0][1]),(line[1][0],line[1][1]),(0,0,255),2)
            # line_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    # draw horizontal lines in yellow
    # for line in validHLC:
    #     cv2.line(line_image,(line[0][0],line[0][1]),(line[1][0],line[1][1]),(0,255,0),2)
    #     line_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)



    # find the closest left line to the center, closest right line to the center, highest horizontal line
    lMax = 0 # rightMost leftLine x coord
    count = 0 # index for loop
    i = 0
    l1M = 0 # first xcoord of lMax
    l2M = 0 # second xcoord of lMax
    l1MH = 0 # first ycoord of lMax
    l2MH = 0 # second ycoord of lMax

    # can find slope, set origin to be where crosses the horizontal line to find y-intercept
    if len(leftLines) > 0:
        for line in leftLines:
            if line[0][0] > lMax:
                lMax = line[0][0]
                i = count
            count = count + 1

        l1M = (leftLines[i])[0][0]
        l2M = (leftLines[i])[1][0]
        l1MH = (leftLines[i])[0][1]
        l2MH = (leftLines[i])[1][1]

        cv2.line(line_image,(l1M,l1MH),(l2M,l2MH),(255,255,0),2)
        line_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)



    # same as lmax
    rMax = frame.shape[1]
    count = 0
    i = 0
    r1M = 0
    r2M = 0
    r1MH = 0
    r2MH = 0

    if len(rightLines) > 0:
        for line in rightLines:
            if line[0][0] < rMax:
                rMax = line[0][0]
                i = count
            count = count + 1

        r1M = (rightLines[i])[0][0]
        r2M = (rightLines[i])[1][0]
        r1MH = (rightLines[i])[0][1]
        r2MH = (rightLines[i])[1][1]

        cv2.line(line_image,(r1M,r1MH),(r2M,r2MH),(255,255,0),2)
        line_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    # same as lmax and rmax 
    hMax = frame.shape[0]
    count = 0
    i = 0
    h1Mx = 0
    h2Mx = 0
    h1My = 0
    h2My = 0
    if len(validHLC) > 0:
        for line in validHLC:
            #print("HORIZONTAL: (",(validHLC[count])[0][0], ",", (validHLC[count])[0][1], ") ", "(",(validHLC[count])[1][0], ",", (validHLC[count])[1][1], ")")
            if line[0][1] < hMax or line[1][1] < hMax:
                hMax = line[0][1]
                i = count
            count = count + 1

        h1Mx = (validHLC[i])[0][0]
        h2Mx = (validHLC[i])[1][0]
        h1My = (validHLC[i])[0][1]
        h2My = (validHLC[i])[1][1]

    cv2.line(line_image,(h1Mx,h1My),(h2Mx,h2My),(0,127,255),2)
    line_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    # at this point have two coordinates for leftmost line and rightmost line
    # can use these and point slope form to derive equation for each line
        
    
    if lMax > 0 and rMax < frame.shape[1] and hMax < frame.shape[0]:

        horizontal_slope = ((float)(h2My-h1My)) / ((float)(h2Mx-h1Mx)) 
        horizontal_int = horizontal_slope*-1*h1Mx+h1My

        print("--")
        print("LEFT: (",l1M, ",", l1MH, ") ", "(",l2M, ",", l2MH, ")")
        print("RIGHT: (",r1M, ",", r1MH, ") ", "(",r2M, ",", r2MH, ")")
        print("HORIZONTAL: (",h1Mx, ",", h1My, ") ", "(",h2Mx, ",", h2My, ")")
        print("--")
        
        # find the slope of each line
        left_slope = ((float)(l2MH-l1MH)) / ((float)(l2M-l1M+.00001)) 
        right_slope = ((float)(r1MH-r2MH)) / ((float)(r1M-r2M+.0001)) 

        # find the y-intercept for each line, y-axis is the left edge of screen, x = 0
        left_int = left_slope*-1*l1M+l1MH
        right_int = right_slope*-1*r2M+r2MH

        print("left slope: ", left_slope)
        print("right slope: ", right_slope)
        print("horizontal slope: ", horizontal_slope)
        print("left int: ", left_int)
        print("right int: ", right_int)
        print("horizontal int: ", horizontal_int)


        # now that we have equations for each line we can get any points on the lines
        # below we provide some y coordinate and can get the according x coordinate to draw horizontal lines

        # this is the bottom line, change it to topmost horizontal line
        #lx = (int)(((float)(cv2.getTrackbarPos('height_low', 'height')-left_int))/left_slope)
        #rx = (int)(((float)(cv2.getTrackbarPos('height_low', 'height')-right_int))/right_slope)
        blcX = (int)(((float)(h1My-left_int))/left_slope)
        brcX = (int)(((float)(h2My-right_int))/right_slope)

        # this is the top line, can use this to find angle
        tlcX = (int)(((float)(cv2.getTrackbarPos('height_high', 'height')-left_int))/left_slope)
        trcX = (int)(((float)(cv2.getTrackbarPos('height_high', 'height')-right_int))/right_slope)

        
        # given some horizontal line we can divide it into three sections
        # dim1 = (int)((rx-lx)/3)
        # dim2 = (int)((r2x-l2x)/3)

        botDim = (int)((brcX-blcX)/3)
        topDim = (int)((trcX-tlcX)/3)

        thetaCross = math.atan(
                    abs(
                        (left_slope - right_slope) / (1 + left_slope*right_slope)
                       )
                              )
                        
        sharedTheta = thetaCross / 6

        dimOfsX = botDim*math.sin(sharedTheta)
        dimOfsY = botDim*math.cos(sharedTheta)

        
        # bottom two points for green lines
        v_gl1_x1 = botDim + blcX
        v_gl2_x1 = botDim*2 + blcX
        v_gl1_y1 = (int)((float)(horizontal_slope*v_gl1_x1+horizontal_int))
        v_gl2_y1 = (int)((float)(horizontal_slope*v_gl2_x1+horizontal_int)) 

        # top two points for green lines
        v_gl1_x2 = topDim + tlcX
        v_gl2_x2 = topDim*2 + tlcX
        v_gl1_y2 = cv2.getTrackbarPos('height_high', 'height')
        v_gl2_y2 = cv2.getTrackbarPos('height_high', 'height')

        cv2.line(line_image,(v_gl1_x1,v_gl1_y1),(v_gl1_x2,v_gl1_y2),(0,255,0),2)
        cv2.line(line_image,(v_gl2_x1,v_gl2_y1),(v_gl2_x2,v_gl2_y2),(0,255,0),2)

        print("Green Bottoms: (",v_gl1_x1, ",", v_gl1_y1, ") ", "(",v_gl2_x1, ",", v_gl2_y1, ")")


        # first horizontal grid line
        h_gl1_yL = (int)(-dimOfsY + v_gl1_y1)
        h_gl1_yR = (int)(-dimOfsY + v_gl2_y1)
        h_gl1_xL = (int)(((float)(h_gl1_yL-left_int))/left_slope)
        h_gl1_xR = (int)(((float)(h_gl1_yR-right_int))/right_slope)
        #h_gl1_xL = (int)(v_gl1_x1 + dimOfsX)
        #h_gl1_xR = (int)(v_gl2_x1 - dimOfsX)
        cv2.line(line_image,(h_gl1_xL,h_gl1_yL),(h_gl1_xR,h_gl1_yR),(0,255,0),2)


        dimOne = (int)((h_gl1_xR - h_gl1_xL)/3)
        dim1OfsY = dimOne*math.cos(sharedTheta)

        # second horizontal grid line
        h_gl2_yL = (int)(-dim1OfsY + h_gl1_yL)
        h_gl2_yR = (int)(-dim1OfsY + h_gl1_yR)
        h_gl2_xL = (int)(((float)(h_gl2_yL-left_int))/left_slope)
        h_gl2_xR = (int)(((float)(h_gl2_yR-right_int))/right_slope)
        cv2.line(line_image,(h_gl2_xL,h_gl2_yL),(h_gl2_xR,h_gl2_yR),(0,255,0),2)

        dimTwo = (int)((h_gl2_xR - h_gl2_xL)/3)
        dim2OfsY = dimTwo*math.cos(sharedTheta)

        # third horizontal grid line
        h_gl3_yL = (int)(-dim2OfsY + h_gl2_yL)
        h_gl3_yR = (int)(-dim2OfsY + h_gl2_yR)
        h_gl3_xL = (int)(((float)(h_gl3_yL-left_int))/left_slope)
        h_gl3_xR = (int)(((float)(h_gl3_yR-right_int))/right_slope)
        cv2.line(line_image,(h_gl3_xL,h_gl3_yL),(h_gl3_xR,h_gl3_yR),(0,255,0),2)

        # to complete the grid we want to draw vertical lines through the sections defined by dim
        # we get the xcoord by adding dim*n to the left line, y is whatever we want it to be
        # glx1 = dim1+lx
        # gly1 = cv2.getTrackbarPos('height_low', 'height')

        # glx2 = dim2+l2x
        # gly2 = cv2.getTrackbarPos('height_high', 'height')

        # gl2x1 = 2*dim1+lx
        # gl2y1 = cv2.getTrackbarPos('height_low', 'height')

        # gl2x2 = 2*dim2+l2x
        # gl2y2 = cv2.getTrackbarPos('height_high', 'height')

        # height = (int)((gly2-gly1)/3)

        # l3x = (int)(((float)(cv2.getTrackbarPos('height_low', 'height')+height-left_int))/left_slope)
        # r3x = (int)(((float)(cv2.getTrackbarPos('height_low', 'height')+height-right_int))/right_slope)

        # l4x = (int)(((float)(cv2.getTrackbarPos('height_low', 'height')+2*height-left_int))/left_slope)
        # r4x = (int)(((float)(cv2.getTrackbarPos('height_low', 'height')+2*height-right_int))/right_slope)

        # l3x = (int)(((float)(100-left_int))/left_slope)
        # r3x = (int)(((float)(100-right_int))/right_slope)

        # cv2.line(line_image,(lx,cv2.getTrackbarPos('height_low', 'height')),(rx,cv2.getTrackbarPos('height_low', 'height')),(0,255,255),2)
        # cv2.line(line_image,(l2x,cv2.getTrackbarPos('height_high', 'height')),(r2x,cv2.getTrackbarPos('height_high', 'height')),(0,255,255),2)

        # # new stuff
        # cv2.line(line_image,(l3x,cv2.getTrackbarPos('height_low', 'height')+height),(r3x,cv2.getTrackbarPos('height_low', 'height')+height),(0,255,255),2)
        # cv2.line(line_image,(l4x,cv2.getTrackbarPos('height_low', 'height')+2*height),(r4x,cv2.getTrackbarPos('height_low', 'height')+2*height),(0,255,255),2)

        # cv2.line(line_image,(glx1,gly1),(glx2,gly2),(0,255,255),2)
        # cv2.line(line_image,(gl2x1,gl2y1),(gl2x2,gl2y2),(0,255,255),2)

        # cv2.line(line_image,(l2x,300),(r2x,300),(0,255,255),2)
        # cv2.line(line_image,(l3x,100),(r3x,100),(0,255,255),2)
        line_edges = cv2.addWeighted(frame, 0.8, line_image, 1, 0)




    cv2.imshow("ROI", masked_edges)
    cv2.imshow("overlay", line_edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
