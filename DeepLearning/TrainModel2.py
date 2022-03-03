"""Importation des bibliothèques nécessaires"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob as glob
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os

"""
Définition des variables importantes :
- Taille de l'image pour l'entrainement
- Chemin vers la base d'entrainement
- Chemin vers la base de validation
- La taille des paquets de données envoyés au modèle pour l'entrainement (nombre d'images)
"""
IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR = './ProjetS4/dataset/train/'
VALID_DATA_DIR = './ProjetS4/dataset/valid/'
batch_size = 32

# redimensionnement des images
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255
)

# chargement des données d'entrainement
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAINING_DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SHAPE,
    batch_size=batch_size)

# chargement des données de validation
val_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SHAPE,
    batch_size=batch_size)

# Enregistrement des noms des classes dans une variable
class_names = train_ds.class_names

# Optimisation du programme via une API de TensorFlow
AUTOTUNE = tf.data.AUTOTUNE

# Prétraitement des données d'entrainement pour améliorer les performances
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Standardisation des données (réécriture des couleurs de 0 à 1 au lieu de 0 à 255)
normalization_layer = layers.Rescaling(1. / 255)

"""
Augmentation des données d'entrainement via des rotations, décalages, zoom, etc réalisés aléatoirement par TensorFlow
"""
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=IMAGE_SHAPE + (3,)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

"""
Définition du modèle de réseau de neurones :
- Utilisation de l'augmentation des données
- Utilisation de la fonction de réseau de neurones "Sequential" qui se compose de trois blocs de convolution :
* Le premier avec 16 filtres avec un kernel de taille 3 (?) sans padding et avec une activation de type "relu" 
(rectified linear unit)
* Le second est le même mais avec 32 filtres
* Le derner est le même mais avec 64 filtres
- Chaque couche a une couche de regroupement maximum (?)
- Il y a une couche avec 128 neurones
- Puis enfin une couche avec 4 neurones pour déterminer les classes
"""


def create_modelClean():
    model = Sequential([
        data_augmentation,
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names))
    ])

    '''
    Compilation du modèle avec l'optimisation "Adam" et la fonction de perte "sparse_categorical_crossentropy", le dernier 
    paramètres permet d'afficher la précision du modèle
    '''
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


model = create_modelClean()

# Affichage de toutes les couches du modèle
model.summary()

# Initialisation de l'entrainement du modèle avec 15 Epochs, la base de validation et d'entrainement
epochs = 1
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=1)

model.save('./ProjetS4/Deeplearning/saved_model/modelBad')

# Enregistrement des résultats de précisions et de pertes
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

"""
Affichage des résultats de précisions et de pertes en fonction des Epochs via MatplotLib
"""

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
