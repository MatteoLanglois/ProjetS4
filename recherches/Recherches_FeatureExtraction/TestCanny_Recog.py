import cv2
import matplotlib.pyplot as plt

img_with = [cv2.imread(f'./dataset/train/with_mask/with_mask_{I}.jpg') for I in range(0, 2)]
img_without = [cv2.imread(f'./dataset/train/without_mask/without_mask_{I}.jpg') for I in range(0, 2)]
img_incorrect = [cv2.imread(f'./dataset/train/incorrect_mask/incorrect_mask_{I}.jpg') for I in [0, 3]]


img_with_edit = []
img_without_edit = []
img_incorrect_edit = []

img_with_edit_rs = []
img_without_edit_rs = []
img_incorrect_edit_rs = []

for I in range(len(img_with)):
    img_with_edit.append(cv2.Canny(img_with[I], 100, 200, apertureSize=3, L2gradient=True))
    img_without_edit.append(cv2.Canny(img_without[I], 100, 200, apertureSize=3, L2gradient=True))
    img_incorrect_edit.append(cv2.Canny(img_incorrect[I], 100, 200, apertureSize=3, L2gradient=True))

    img_with_edit_rs.append(cv2.resize(img_with_edit[I], (600, 900), interpolation=cv2.INTER_CUBIC))
    img_without_edit_rs.append(cv2.resize(img_without_edit[I], (600, 900), interpolation=cv2.INTER_CUBIC))
    img_incorrect_edit_rs.append(cv2.resize(img_incorrect_edit[I], (600, 900), interpolation=cv2.INTER_CUBIC))

    plt.subplot(3, 3, I + 1)
    plt.imshow(img_with_edit_rs[I])
    plt.subplot(3, 3, I + 4)
    plt.imshow(img_without_edit_rs[I])
    plt.subplot(3, 3, I + 7)
    plt.imshow(img_incorrect_edit_rs[I])

plt.show()
