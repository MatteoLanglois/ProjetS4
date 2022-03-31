import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR = './dataset/train/'

train_ds = tf.keras.utils.image_dataset_from_directory(TRAINING_DATA_DIR)

class_names = train_ds.class_names

model = tf.keras.models.load_model('./Deeplearning/saved_model/modelClean')


def MaskDetection(frame):
    # Prétraitement de l'image
    frame = cv2.resize(frame, IMAGE_SHAPE, interpolation=cv2.INTER_CUBIC)
    img_array = tf.keras.utils.img_to_array(frame)
    img_array = tf.expand_dims(img_array, 0)
    # Prédiction de la classe de l'image
    ia_prediction = model.predict(img_array)
    # Récupération de la classe prédite
    class_Prediction = f"{class_names[np.argmax(tf.nn.softmax(ia_prediction[0]))]} : {round(np.max(tf.nn.softmax(ia_prediction[0])) * 100, 1)}%"

    return class_Prediction


cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    _, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        prediction = MaskDetection(img[y - 10:y + h + 10, x - 10:x + w + 10])
        cv2.putText(img, prediction, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit

cv2.destroyAllWindows()
