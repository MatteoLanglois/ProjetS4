import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

modelc = Sequential([
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(axis=1),
    layers.DepthwiseConv2D(3, padding='same', activation='relu'),
    layers.BatchNormalization(axis=1),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(axis=1),
    layers.DepthwiseConv2D(3, padding='same', activation='relu'),
    layers.BatchNormalization(axis=1),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(axis=1),
    layers.DepthwiseConv2D(3, padding='same', activation='relu'),
    layers.BatchNormalization(axis=1),
])

modelc.compile(optimizer='Adam',
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])

modelc.build(input_shape=(1, 224, 224, 3))


modelc.summary()

baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=1e-4, decay=1e-4 / 25)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])
