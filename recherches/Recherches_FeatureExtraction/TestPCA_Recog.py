import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import glob

pca = PCA(235)

image_paths = [I for I in glob.glob("./input/*.jpg")]
print(f"Found {len(image_paths)} images...")
plt.figure(figsize=(16, 12))

plt.axis('off')
for i, image_path in enumerate(image_paths):
    orig_image = plt.imread(image_path)
    red, green, blue = cv2.split(orig_image)

    red_invert = pca.inverse_transform(pca.fit_transform(red))
    green_invert = pca.inverse_transform(pca.fit_transform(green))
    blue_invert = pca.inverse_transform(pca.fit_transform(blue))

    img_compressed = (np.dstack((red_invert, green_invert, blue_invert))).astype(np.uint8)

    plt.subplot(5, 10, 2 * i + 1)
    plt.imshow(img_compressed)
    plt.axis('off')

plt.show()
