# importing the dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

img = cv2.imread('ProjetS4/recherches/images/Nana.jpg')
blue, green, red = cv2.split(img)

pca = PCA(200)

# Applying to red channel and then applying inverse transform to transformed array.
red_transformed = pca.fit_transform(red)
red_inverted = pca.inverse_transform(red_transformed)

# Applying to Green channel and then applying inverse transform to transformed array.
green_transformed = pca.fit_transform(green)
green_inverted = pca.inverse_transform(green_transformed)

# Applying to Blue channel and then applying inverse transform to transformed array.
blue_transformed = pca.fit_transform(blue)
blue_inverted = pca.inverse_transform(blue_transformed)

img_compressed = (np.dstack((red_inverted, red_inverted, red_inverted))).astype(np.uint8)
plt.imshow(img_compressed)
plt.show()

'''
digits = load_digits()
data = digits.data
data.shape

image_sample = data[0,:].reshape(8,8)
plt.imshow(image_sample)

pca = PCA(2)  # we need 2 principal components.
converted_data = pca.fit_transform(digits.data)

converted_data.shape

plt.style.use('seaborn-whitegrid')
plt.figure(figsize = (10,6))
c_map = plt.cm.get_cmap('jet', 10)
plt.scatter(converted_data[:, 0], converted_data[:, 1], s = 15,
            cmap = c_map , c = digits.target)
plt.colorbar()
plt.xlabel('PC-1') , plt.ylabel('PC-2')
plt.show()
'''