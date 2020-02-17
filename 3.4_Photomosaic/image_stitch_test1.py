import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# perform colour dominant tests
# 2 long colours
# 6 short colours
# image with 4 colours is on top
# second image with 1 long colour is bottom,
# bottom box's long colour touches the matching side of the top box

# get two small boxes, match to correct side of bottom box
# align remaining long box to the right of the right small box


"""
    convert each of the five faces of the box into arrays 
"""

folders = glob.glob('*')
print(folders)
imagenames_list = []
for f in folders:
    if '.png' in f:
        imagenames_list.append(f)

read_images = []
print(imagenames_list)

for image in imagenames_list:
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    read_images.append(img)

"""

"""