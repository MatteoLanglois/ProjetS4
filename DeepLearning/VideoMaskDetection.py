"""Importation des bibliothèques nécessaires"""
import cv2
import tensorflow as tf
import numpy as np

"""
Définition des variables importantes :
- Taille de l'image
- Chemin vers la base d'entrainement
"""
IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR = './dataset/train/'

# Enregistrement des noms des classes dans une variable
train_ds = tf.keras.utils.image_dataset_from_directory(TRAINING_DATA_DIR)
class_names = train_ds.class_names

# Récupération du modèle
model = tf.keras.models.load_model('./Deeplearning/saved_model/modelClean')


def MaskDetection(frame):
    # Prétraitement de l'image
    frame = cv2.resize(frame, IMAGE_SHAPE, interpolation=cv2.INTER_CUBIC)
    img_array = tf.keras.utils.img_to_array(frame)
    img_array = tf.expand_dims(img_array, 0)
    # Prédiction de la classe de l'image
    ia_prediction = model.predict(img_array)
    # Récupération de la classe prédite et de la probabilité
    return f"{class_names[np.argmax(tf.nn.softmax(ia_prediction[0]))]} : " \
           f"{round(np.max(tf.nn.softmax(ia_prediction[0])) * 100, 1)}% "


# Récupération de la vidéo
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    # Récupération de l'image
    _, img = cam.read()
    # Récupération du classifier pour la détection de visage
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Détection des visages
    faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.3, 5)
    # Pour chaque visage
    for (x, y, w, h) in faces:
        # Dessin du rectangle autour du visage
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # Prédiction de la classe du visage
        prediction = MaskDetection(img[y - 10:y + h + 10, x - 10:x + w + 10])
        # Affichage du résultat
        cv2.putText(img, prediction, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # Affichage de l'image
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit

cv2.destroyAllWindows()
