import cv2
import numpy as np
import matplotlib.pyplot as plt
 
if __name__ == '__main__' :

    
    img1 = cv2.imread('coral1.jpg',0)
    img2 = cv2.imread('coral2.jpg',0)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

# Initialize lists
    list_kp1 = []
    list_kp2 = []

# For each match...
    for mat in matches:

    # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

    # x - columns
    # y - rows
    # Get the coordinates
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

    # Append to each list
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))

#these two lists will contain all the coordinates of the feature points
    list_kp1 = [kp1[mat.queryIdx].pt for mat in matches] 
    list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]

#test print
    for p in list_kp1: print (p)

    
    
    # Read source image.
    im_src = cv2.imread('coral2.jpg')
    # Four corners of the book in source image
    pts_src =np.array([list_kp2[0],list_kp2[1],list_kp2[2],list_kp2[3]])
 
 
    # Read destination image.
    im_dst = cv2.imread('coral1.jpg')
    # Four corners of the book in destination image.
    pts_dst = np.array([list_kp1[0], list_kp1[1], list_kp1[2],list_kp1[3]])
 
    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)
     
    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
     
    # Display images
    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)
    cv2.imshow("Warped Source Image", im_out)
 
    cv2.waitKey(0)
