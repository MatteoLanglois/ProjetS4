import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_pixel(img, center, x, y):
    new_value = 0
    try:
        # If local neighbourhood pixel
        # value is greater than or equal
        # to center pixel values then
        # set it to 1
        if img[x][y] >= center:
            new_value = 1
    except:
        # Exception is required when
        # neighbourhood value of a center
        # pixel value is null i.e. values
        # present at boundaries.
        pass

    return new_value


# Function for calculating LBP
def lbp_calculated_pixel(img, x, y):
    center = img[x][y]

    val_ar = [get_pixel(img, center, x - 1, y - 1), get_pixel(img, center, x - 1, y),
              get_pixel(img, center, x - 1, y + 1), get_pixel(img, center, x, y + 1),
              get_pixel(img, center, x + 1, y + 1), get_pixel(img, center, x + 1, y),
              get_pixel(img, center, x + 1, y - 1), get_pixel(img, center, x, y - 1)]

    # Now, we need to convert binary
    # values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]

    val = 0

    for val_i in range(len(val_ar)):
        val += val_ar[val_i] * power_val[val_i]

    return val


img_with = [cv2.imread(f'ProjetS4/dataset/train/with_mask/with_mask_{I}.jpg') for I in range(0, 2)]
img_without = [cv2.imread(f'ProjetS4/dataset/train/without_mask/without_mask_{I}.jpg') for I in range(0, 2)]
img_incorrect = [cv2.imread(f'ProjetS4/dataset/train/incorrect_mask/incorrect_mask_{I}.jpg') for I in [0, 3]]

img_with_edit = []
img_without_edit = []
img_incorrect_edit = []

for I in range(len(img_with)):
    height_with, width_with, _ = img_with[I].shape
    height_without, width_without, _ = img_without[I].shape
    height_incorrect, width_incorrect, _ = img_incorrect[I].shape

    # We need to convert RGB image
    # into gray one because gray
    # image has one channel only.
    img_with_edit.append(cv2.cvtColor(img_with[I], cv2.COLOR_BGR2GRAY))
    img_without_edit.append(cv2.cvtColor(img_without[I], cv2.COLOR_BGR2GRAY))
    img_incorrect_edit.append(cv2.cvtColor(img_incorrect[I], cv2.COLOR_BGR2GRAY))

    # Create a numpy array as
    # the same height and width
    # of RGB image
    img_lbp_with = np.zeros((height_with, width_with), np.uint8)
    img_lbp_without = np.zeros((height_without, width_without), np.uint8)
    img_lbp_incorrect = np.zeros((height_incorrect, width_incorrect), np.uint8)

    # Calculate LBP for each pixel
    for i in range(0, height_with):
        for j in range(0, width_with):
            img_lbp_with[i][j] = lbp_calculated_pixel(img_with_edit[I], i, j)

    for i in range(0, height_without):
        for j in range(0, width_without):
            img_lbp_without[i][j] = lbp_calculated_pixel(img_without_edit[I], i, j)

    for i in range(0, height_incorrect):
        for j in range(0, width_incorrect):
            img_lbp_incorrect[i][j] = lbp_calculated_pixel(img_incorrect_edit[I], i, j)

    plt.subplot(3, 3, I + 1)
    plt.imshow(img_lbp_with, cmap='gray')
    plt.subplot(3, 3, I + 4)
    plt.imshow(img_lbp_without, cmap='gray')
    plt.subplot(3, 3, I + 7)
    plt.imshow(img_lbp_incorrect, cmap='gray')

plt.show()

print("LBP Program is finished")
