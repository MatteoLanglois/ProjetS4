"""Importation des bibliothèques nécessaires"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob as glob
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
import pandas as pd
import seaborn as sn

"""
Définition des variables importantes :
- Taille de l'image
- Chemin vers la base d'entrainement
- Matrice de confusion
"""
IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR = './ProjetS4/dataset/train/'
MatriceConf = {"Y actuel": [], "Y prédiction": []}

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

model = tf.keras.models.load_model('./ProjetS4/Deeplearning/saved_model/modelClean')

image_paths = glob.glob('./ProjetS4/input/*.jpg')
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
for i, image_path in enumerate(image_paths):
    orig_image = plt.imread(image_path)
    plt.subplot(5, 10, 2 * i + 1)
    plt.imshow(orig_image)
    plt.axis('off')
    plt.title(f"{class_names[np.argmax(predictions[i][1])]} ({round(np.max(predictions[i][1]) * 100, 2)}%)")
    plt.subplot(5, 10, 2 * i + 2)
    plt.pie(predictions[i][1].numpy() * 100, colors=["#37B0B3", "#4453B3", "#BF46F0"])
    nameIMG = image_path.replace("./ProjetS4/input", "")
    MatriceConf["Y actuel"].append(class_names.index(nameIMG[1:-6]))
    MatriceConf["Y prédiction"].append(np.argmax(predictions[i][1]))

plt.subplot(5, 10, 35)
df = pd.DataFrame(MatriceConf, columns=['Y actuel','Y prédiction'])
confusion_matrix = pd.crosstab(df['Y actuel'], df['Y prédiction'], rownames=['actuel'], colnames=['prédiction'], margins = True)
sn.heatmap(confusion_matrix, annot=True)

plt.subplot(5, 10, 36)
plt.axis("off")
plt.text(0.2, 0.6, f"0 : {class_names[0]}", fontsize=12)
plt.text(0.2, 0.3, f"1 : {class_names[1]}", fontsize=12)
plt.text(0.2, 0, f"2 : {class_names[2]}", fontsize=12)

accuracy = round((confusion_matrix.iloc[0,0] + confusion_matrix.iloc[1,1] + confusion_matrix.iloc[2,2]) / len(image_paths), 3)*100
plt.subplot(5, 10, 45)
plt.axis('off')
plt.text(-0.8, 0.4, f"Précision totale: {accuracy}%", fontsize=20)


# Affichage des différents graphiques matplotlib
plt.legend(class_names, loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()
