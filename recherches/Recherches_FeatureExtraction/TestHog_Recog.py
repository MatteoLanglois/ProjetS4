import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure, data
import matplotlib.pyplot as plt
import cv2

img_with = [imread(f'./dataset/train/with_mask/with_mask_{I}.jpg') for I in range(0, 2)]
img_without = [imread(f'./dataset/train/without_mask/without_mask_{I}.jpg') for I in range(0, 2)]
img_incorrect = [imread(f'./dataset/train/incorrect_mask/incorrect_mask_{I}.jpg') for I in [0, 3]]


img_with_edit = ["" for I in range(len(img_with))]
img_without_edit = ["" for I in range(len(img_with))]
img_incorrect_edit = ["" for I in range(len(img_with))]

img_with_edit_rs = []
img_without_edit_rs = []
img_incorrect_edit_rs = []

for I in range(len(img_with)):
    fd, img_with_edit = hog(img_with[I], orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    fd, img_without_edit = hog(img_without[I], orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    fd, img_incorrect_edit = hog(img_incorrect[I], orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, channel_axis=-1)

    img_with_edit_rs.append(exposure.rescale_intensity(img_with[I], in_range=(0, 10)))
    img_without_edit_rs.append(exposure.rescale_intensity(img_without[I], in_range=(0, 10)))
    img_incorrect_edit_rs.append(exposure.rescale_intensity(img_incorrect[I], in_range=(0, 10)))

    plt.subplot(3, 3, I + 1)
    plt.imshow(img_with_edit_rs[I], cmap=plt.cm.gray)
    plt.title(f"With mask {I}")
    plt.subplot(3, 3, I + 4)
    plt.imshow(img_without_edit_rs[I], cmap=plt.cm.gray)
    plt.title(f"Without mask {I}")
    plt.subplot(3, 3, I + 7)
    plt.imshow(img_incorrect_edit_rs[I], cmap=plt.cm.gray)
    plt.title(f"Incorrect mask {I}")

plt.show()
