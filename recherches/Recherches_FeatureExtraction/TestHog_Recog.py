import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure, data
import matplotlib.pyplot as plt
import cv2
import glob

image_paths = [I for I in glob.glob("./input/*.jpg")]
print(f"Found {len(image_paths)} images...")
plt.figure(figsize=(16, 12))

plt.axis('off')
for i, image_path in enumerate(image_paths):
    orig_image = plt.imread(image_path)
    fd, img = hog(orig_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,
              channel_axis=-1)

    img = exposure.rescale_intensity(img, in_range=(0, 10))

    plt.subplot(5, 10, 2 * i + 1)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.axis('off')

plt.show()
