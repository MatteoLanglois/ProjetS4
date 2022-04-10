"""Importation des bibliothèques nécessaires"""
import matplotlib.pyplot as plt
import tensorflow as tf
import glob as glob
import numpy as np
import pandas as pd
import seaborn as sn


"""
Définition des variables importantes :
- Taille de l'image
- Chemin vers la base d'entrainement
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
- Récupération du modèle
- Récupération des images pour le test et affichage du nombre d'images
- Création d'un dictionnaire pour sauvegarder les résultats
"""

model = tf.keras.models.load_model('./Deeplearning/saved_model/modelClean')

image_paths = glob.glob('./dataset/test/*/*.jpg')
print(f"Found {len(image_paths)} images...")
predictions = {}
MatriceConf = {"Y actuel": [], "Y prédiction": []}
acc_w = 0


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
    predictions[i] = [prediction[0], tf.nn.softmax(prediction[0])]
    # Récupération de la classe prédite
    if "incorrect" in image_path:
        MatriceConf["Y actuel"].append(0)
        acc_w += predictions[i][1].numpy()[0]
    elif "without" in image_path:
        MatriceConf["Y actuel"].append(2)
        acc_w += predictions[i][1].numpy()[2]
    else:
        MatriceConf["Y actuel"].append(1)
        acc_w += predictions[i][1].numpy()[1]
    MatriceConf["Y prédiction"].append(np.argmax(prediction[0]))

plt.subplot(3, 2, 6)
plt.axis('off')
df = pd.DataFrame(MatriceConf, columns=['Y actuel', 'Y prédiction'])
confusion_matrix = pd.crosstab(df['Y actuel'], df['Y prédiction'], rownames=['actuel'], colnames=['prédiction'],
                               margins=True)
sn.heatmap(confusion_matrix, annot=True)

accuracy = sum([MatriceConf['Y actuel'][i] == MatriceConf['Y prédiction'][i] for i in range(0, len(image_paths))]) / len(image_paths) * 100
# Calcul d'une précision pondérée

accuracy_weighted = round(acc_w / len(image_paths), 3) * 100


plt.subplot(3, 2, 4)
plt.axis('off')
plt.text(0, 0.4, f"Précision globale: {round(accuracy, 3)}%", fontsize=20)
plt.text(-0.8, 0.1, f"Précision pondérée: {round(accuracy_weighted, 3)}%", fontsize=20)

plt.show()