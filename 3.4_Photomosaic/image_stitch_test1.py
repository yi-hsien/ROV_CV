import cv2
import numpy as np
import matplotlib.pyplot as plt

# perform colour dominant tests
# 2 long colours
# 6 short colours
# image with 4 colours is on top
# second image with 1 long colour is bottom,
# bottom box's long colour touches the matching side of the top box

# get two small boxes, match to correct side of bottom 