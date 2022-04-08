"load model, base test, prédictions, matrice confusion"

import matplotlib.pyplot as plt
import tensorflow as tf
import glob as glob

#model = tf.keras.models.load_model('./Deeplearning/saved_model/modelClean')

image_paths = glob.glob('./dataset/test/*/*.jpg')
print(f"Found {len(image_paths)} images...")
plt.figure(figsize=(16, 12))

for i, image_path in enumerate(image_paths):
    chemin=plt.imread(image_paths)
    img = tf.keras.utils.load_img(
        image_path, target_size=IMAGE_SHAPE
    )