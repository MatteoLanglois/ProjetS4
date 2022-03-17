import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob


def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

# Function for calculating LBP
def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = [get_pixel(img, center, x - 1, y - 1), get_pixel(img, center, x - 1, y),
              get_pixel(img, center, x - 1, y + 1), get_pixel(img, center, x, y + 1),
              get_pixel(img, center, x + 1, y + 1), get_pixel(img, center, x + 1, y),
              get_pixel(img, center, x + 1, y - 1), get_pixel(img, center, x, y - 1)]
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for val_i in range(len(val_ar)):
        val += val_ar[val_i] * power_val[val_i]
    return val


image_paths = [I for I in glob.glob("./input/*.jpg")]
print(f"Found {len(image_paths)} images...")
plt.figure(figsize=(16, 12))

plt.axis('off')
for iter, image_path in enumerate(image_paths):
    orig_image = plt.imread(image_path)
    height_with, width_with, _ = orig_image.shape

    img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    img_zero = np.zeros((height_with, width_with), np.uint8)

    for i in range(0, height_with):
        for j in range(0, width_with):
            img_zero[i][j] = lbp_calculated_pixel(img, i, j)

    plt.subplot(5, 10, 2 * iter + 1)
    plt.imshow(img_zero, cmap=plt.cm.gray)
    plt.axis('off')

plt.show()

print("LBP Program is finished")
