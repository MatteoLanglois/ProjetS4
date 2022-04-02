"""Importation des bibliothèques nécessaires"""
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

"""
Définition des variables importantes :
- Taille de l'image pour l'entrainement
- Chemin vers la base d'entrainement
- Chemin vers la base de validation
- La taille des paquets de données envoyés au modèle pour l'entrainement (nombre d'images)
"""
IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR = './dataset/train/'
VALID_DATA_DIR = './dataset/valid/'
batch_size = 32

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
avec : 
- Mise en miroir aléatoire
- Rotation aléatoire
- Zoom aléatoire
- Changement de la taille de l'image aléatoirement
- Changement de l'échelle de l'image aléatoirement
- Changement de la luminosité de l'image aléatoirement
"""
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=IMAGE_SHAPE + (3,)),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.Resizing(180, 180),
        layers.Rescaling(1. / 255),
        layers.RandomContrast((0.2, 1.2)),

    ]
)

"""
Définition du modèle de réseau de neurones :
- Utilisation de l'augmentation des données
- Utilisation de la fonction de réseau de neurones "Sequential" qui se compose de plusieurs couches de neurones :
* La première est une couche dense avec 16 neurones
* La deuxième est une couche avec 16 neurones convolutifs
* La troisième est une couche avec 32 neurones convolutifs
* La quatrième est une couche avec 32 neurones convolutifs
* La cinquième est une couche avec 64 neurones convolutifs
* La sixième est une couche avec 64 neurones convolutifs
* La septième est une couche avec 128 neurones convolutifs
* La huitième est une couche avec 3 neurones de profondeur (?)
* La neuvième est une couche denses avec 128 neurones
- Puis enfin une couche avec 3 neurones pour déterminer les classes
"""


def create_modelClean():
    model_c = Sequential([
        data_augmentation,
        layers.Dense(16, activation='relu'),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.BatchNormalization(axis=1),
        layers.DepthwiseConv2D(3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names))
    ])
    '''Compilation du modèle avec l'optimisation "Adam" et la fonction de perte "sparse_categorical_crossentropy", 
    le dernier paramètres permet d'améliorer le modèle en fonction de la précision '''
    model_c.compile(optimizer='Adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    return model_c


# Création du modèle
model = create_modelClean()
# Affichage de toutes les couches du modèle
model.summary()
# Initialisation de l'entrainement du modèle avec 30 Epochs, la base de validation et d'entrainement
epochs = 30
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=1)

# Sauvegarde du modèle
model.save('./Deeplearning/saved_model/modelClean')

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
plt.title(f'Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title(f'Training and Validation Loss')
plt.show()
