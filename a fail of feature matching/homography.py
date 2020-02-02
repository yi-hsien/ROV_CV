import cv2
import numpy as np
 
if __name__ == '__main__' :
 
    # Read source image.
    im_src = cv2.imread('coral2.jpg')
    # Four corners of the book in source image
    pts_src =np.array([[19,175],[155,174],[163,149],[18,142]])
 
 
    # Read destination image.
    im_dst = cv2.imread('coral1.jpg')
    # Four corners of the book in destination image.
    pts_dst = np.array([[10, 176], [181, 173], [195,137],[11,137]])
 
    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)
     
    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
     
    # Display images
    cv2.imshow("Source Image", im_src)
    cv2.imshow("Destination Image", im_dst)
    cv2.imshow("Warped Source Image", im_out)
 
    cv2.waitKey(0)
