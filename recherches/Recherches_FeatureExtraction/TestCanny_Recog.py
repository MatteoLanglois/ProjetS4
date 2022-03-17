import cv2
import matplotlib.pyplot as plt
import glob

image_paths = [I for I in glob.glob("./input/*.jpg")]
print(f"Found {len(image_paths)} images...")
plt.figure(figsize=(16, 12))

plt.axis('off')
for i, image_path in enumerate(image_paths):
    orig_image = plt.imread(image_path)
    img = cv2.Canny(orig_image, 100, 200, apertureSize=3, L2gradient=True)

    plt.subplot(5, 10, 2 * i + 1)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.axis('off')

plt.show()
