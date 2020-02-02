import numpy as np
import cv2
import matplotlib.pyplot as plt

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



#tranformation starting here
img = cv2.imread('coral1.jpg')
rows,cols,ch = img.shape

#put for feature points
pts1 = np.float32([list_kp1[0],list_kp1[1],list_kp1[2],list_kp1[3]])
pts2 = np.float32([list_kp2[0],list_kp2[1],list_kp2[2],list_kp2[3]])

#transformation matrix
M = cv2.getPerspectiveTransform(pts1,pts2)

#apply matrix
dst = cv2.warpPerspective(img,M,(200,200))


##create image
cv2.imwrite('c:\Python\coral3.jpg',dst)


#print out matrix
plt.subplot(221),plt.imshow(img),plt.title('Input')
plt.subplot(222),plt.imshow(dst),plt.title('Output')
plt.subplot(224),plt.imshow(img2),plt.title('compare')
plt.show()






#print two images together with feature points matched
#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:4],None, flags=2)
#plt.imshow(img3)
#plt.show()

