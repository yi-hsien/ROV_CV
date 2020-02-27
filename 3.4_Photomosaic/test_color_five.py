import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


folders = glob.glob('*')
print(folders)
imagenames_list = []
for f in folders:
    if '.png' in f:
        imagenames_list.append(f)

print(imagenames_list)

domList = []


for p in imagenames_list:
    img = cv2.imread(p)      # take image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
    average = img.mean(axis=0).mean(axis=0)
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 7    # detects dominant colours
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]

    # matplotlib

    # avg_patch = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(average)

    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
    rows = np.int_(img.shape[0]*freqs)

    dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)

    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
    domList.append(dom_patch)
##    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,6))
##    ax0.imshow(avg_patch)
##    ax0.set_title('Average color')
##    ax0.axis('off')
##    ax1.imshow(dom_patch)
##    ax1.set_title('Dominant colors')
##    ax1.axis('off')
##    fig.show()

fig2, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, figsize=(12, 6))
ax0.imshow(domList[0])
ax1.imshow(domList[1])
ax2.imshow(domList[2])
ax3.imshow(domList[3])
ax4.imshow(domList[4])
fig2.show()

# add to the list of face colours the list of most common rgb values from each face

yuh_list = []

print(domList[0][130][0])
yuh_list.append(domList[0][0][0])
print(yuh_list)
for i in range(len(domList)):
    true_unique_rgbs = []
    print("Printing unique RGB for side:" + str(i))
    for j in range(len(domList[i])):
        for k in range(len(domList[i][j])):
            if domList[i][j][k].any() in true_unique_rgbs:
                true_unique_rgbs.append(domList[i][j][k])
                print(true_unique_rgbs)
            else:
                continue
    print(true_unique_rgbs)
