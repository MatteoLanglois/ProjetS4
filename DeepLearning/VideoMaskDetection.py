import cv2
import imutils
import tensorflow as tf
import numpy as np

IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR = './ProjetS4/dataset/train/'

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAINING_DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SHAPE,
    batch_size=32)

class_names = train_ds.class_names

model = tf.keras.models.load_model('./ProjetS4/Deeplearning/saved_model/modelClean')


def MaskDetection(frame):
    frame = imutils.resize(frame, width=224)
    img_array = tf.keras.utils.img_to_array(frame)
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    class_name = class_names[np.argmax(prediction[0])]
    return class_name


cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret_val, img = cam.read()
    cv2.imshow('Webcam', img)
    img = cv2.resize(img, IMAGE_SHAPE)
    predic = MaskDetection(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"-------\n{predic}\n-------")
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
