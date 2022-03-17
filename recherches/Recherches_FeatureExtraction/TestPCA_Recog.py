import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(180)

img_with = [cv2.imread(f'./dataset/train/with_mask/with_mask_{I}.jpg') for I in range(0, 2)]
img_without = [cv2.imread(f'./dataset/train/without_mask/without_mask_{I}.jpg') for I in range(0, 2)]
img_incorrect = [cv2.imread(f'./dataset/train/incorrect_mask/incorrect_mask_{I}.jpg') for I in [0, 3]]

img_with_edit = []
img_without_edit = []
img_incorrect_edit = []

for I in range(len(img_with)):
    with_red, with_green, with_blue = cv2.split(img_with[I])
    without_red, without_green, without_blue = cv2.split(img_without[I])
    incorrect_red, incorrect_green, incorrect_blue = cv2.split(img_incorrect[I])

    red_invert_with = pca.inverse_transform(pca.fit_transform(with_red))
    red_invert_without = pca.inverse_transform(pca.fit_transform(without_red))
    red_invert_incorrect = pca.inverse_transform(pca.fit_transform(incorrect_red))

    green_invert_with = pca.inverse_transform(pca.fit_transform(with_green))
    green_invert_without = pca.inverse_transform(pca.fit_transform(without_green))
    green_invert_incorrect = pca.inverse_transform(pca.fit_transform(incorrect_green))

    blue_invert_with = pca.inverse_transform(pca.fit_transform(with_blue))
    blue_invert_without = pca.inverse_transform(pca.fit_transform(without_blue))
    blue_invert_incorrect = pca.inverse_transform(pca.fit_transform(incorrect_blue))

    img_compressed_with = (np.dstack((red_invert_with, green_invert_with, blue_invert_with))).astype(np.uint8)
    img_compressed_without = (np.dstack((red_invert_without, green_invert_without, blue_invert_without))).astype(np.uint8)
    img_compressed_incorrect = (np.dstack((red_invert_incorrect, green_invert_incorrect, blue_invert_incorrect))).astype(np.uint8)

    plt.subplot(3, 3, I + 1)
    plt.imshow(img_compressed_with)
    plt.subplot(3, 3, I + 4)
    plt.imshow(img_compressed_without)
    plt.subplot(3, 3, I + 7)
    plt.imshow(img_compressed_incorrect)

plt.show()
