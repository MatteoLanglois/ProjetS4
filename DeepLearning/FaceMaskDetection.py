"""Importation des bibliothèques nécessaires"""
import matplotlib.pyplot as plt
import tensorflow as tf
import glob as glob
from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

import projets4.utils.show as show

"""
Définition des variables importantes :
- Taille de l'image
- Chemin vers la base d'entrainement
- Matrice de confusion
"""
IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR = './dataset/train/'


# chargement des données d'entrainement (pour récupérer les noms des classes)
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAINING_DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SHAPE,
    batch_size=32)

# Enregistrement des noms des classes dans une variable
class_names = train_ds.class_names

"""
Test du modèle avec d'autres données :
- Récupération des images pour le test et affichage du nombre d'images
"""

model = tf.keras.models.load_model('./Deeplearning/saved_model/modelClean')

image_paths = glob.glob('./input/*.jpg')
print(f"Found {len(image_paths)} images...")
plt.figure(figsize=(16, 12))

predictions = {}

for i, image_path in enumerate(image_paths):
    # Enregistrement de l'image dans une variable pour garder une version RGB
    orig_image = plt.imread(image_path)
    # Lecture et prétraitement de l'image (redimensionnement)
    img = tf.keras.utils.load_img(
        image_path, target_size=IMAGE_SHAPE
    )

    # Transformation de l'image en un tableau de données
    img_array = tf.keras.utils.img_to_array(img)
    # Création d'un packets de données
    img_array = tf.expand_dims(img_array, 0)
    # Prédiction de la classe de l'image
    prediction = model.predict(img_array)
    # Récupération de la classe prédite
    predictions[i] = [prediction[0], tf.nn.softmax(prediction[0])]

# Affichage des différentes images ainsi que de la probabilité de prédiction via matplotlib

show.show(image_paths, predictions, class_names, "DL")

# Affichage des différents graphiques matplotlib
plt.show()
