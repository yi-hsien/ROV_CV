import cv2
import numpy as np
import matplotlib.pyplot as plt

<<<<<<< HEAD
img = cv2.imread('face4.png')      # take image
=======
img = cv2.imread('face1.png')      # take image
>>>>>>> 99522edad66f9e2aa5638071901099159090f730
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
average = img.mean(axis=0).mean(axis=0)
pixels = np.float32(img.reshape(-1, 3))

<<<<<<< HEAD
n_colors = 10;    # detects 3 dominant colours
=======
n_colors = 7    # detects 3 dominant colours
>>>>>>> 99522edad66f9e2aa5638071901099159090f730
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
_, counts = np.unique(labels, return_counts=True)
dominant = palette[np.argmax(counts)]

# matplotlib

avg_patch = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(average)

indices = np.argsort(counts)[::-1]
freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
rows = np.int_(img.shape[0]*freqs)

dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
for i in range(len(rows) - 1):
    dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,6))
ax0.imshow(avg_patch)
ax0.set_title('Average color')
ax0.axis('off')
ax1.imshow(dom_patch)
ax1.set_title('Dominant colors')
ax1.axis('off')
fig.show()
