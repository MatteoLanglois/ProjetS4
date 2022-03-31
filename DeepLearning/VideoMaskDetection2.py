"""Importation des bibliothèques nécessaires"""
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils

"""
Définition des variables importantes :
- Taille de l'image pour l'entrainement
- Chemin vers la base d'entrainement$
"""
IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR = './dataset/train/'

# Récupération des noms des classes
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
    # Récupération de la classe prédite
    return f"{class_names[np.argmax(tf.nn.softmax(ia_prediction[0]))]} : " \
           f"{round(np.max(tf.nn.softmax(ia_prediction[0])) * 100, 1)}% "


# Capture de la vidéo
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Récupération des détecteurs pour le visage
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./recherches/Recherches_FeatureExtraction/data/shape_predictor_68_face_landmarks.dat")

while True:
    # Récupération de l'image
    _, img = cam.read()
    # Détection des visages
    img_rects = detector(img, 1)
    for (i, rect) in enumerate(img_rects):
        # Prédiction de la position du visage
        shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), rect)
        shape = face_utils.shape_to_np(shape)
        print(len(shape))
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # Dessin du rectangle autour du visage
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # Affichage du résultat
        cv2.putText(img, MaskDetection(img[y:y + h, x:x + w]), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Affichage de l'image
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit

cv2.destroyAllWindows()
