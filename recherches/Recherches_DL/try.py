import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import argparse

CATEGORIES = ["mask", "no_mask", "incorect_mask"]

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='resnet50',
                    choices=['resnet50', 'vgg16', 'mobilenet'])
args = vars(parser.parse_args())

models_dict = {
    'resnet50': tf.keras.applications.resnet50.ResNet50(weights='imagenet'),
    'vgg16': tf.keras.applications.vgg16.VGG16(weights='imagenet'),
    'mobilenet': tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet'),
}
print(f"Using {args['model']} model...")
# get all the image paths
image_paths = glob.glob('D:/mamac/Documents/Cours/Polytech/Peip2/ProjetS4/testDL/input/*.jpg')
print(f"Found {len(image_paths)} images...")

for i, image_path in enumerate(image_paths):
    print(f"Processing and classifying on {image_path.split('/')[-1]}")
    # read image using matplotlib to keep an original RGB copy
    orig_image = plt.imread(image_path)
    # read and resize the image
    image = tf.keras.preprocessing.image.load_img(image_path,
                                                  target_size=(224, 224))
    # add batch dimension
    image = np.expand_dims(image, axis=0)
    # preprocess the image using TensorFlow utils
    image = tf.keras.applications.imagenet_utils.preprocess_input(image)
    # load the model
    model = models_dict[args['model']]
    # forward pass through the model to get the predictions
    predictions = model.predict(image)
    processed_preds = tf.keras.applications.imagenet_utils.decode_predictions(
        preds=predictions
    )
    print(f"Original predictions: {predictions}")
    print(np.argmax(predictions[0]))
    print(f"Processed predictions: {CATEGORIES[np.argmax(predictions[0][0])]}")
    print('-' * 50)

    # create subplot of all images
    plt.subplot(4, 7, i+1)
    plt.imshow(orig_image)
    plt.title(f"{CATEGORIES[np.argmax(predictions[0][0])]}")
    plt.axis('off')
# plt.savefig(f"outputs/{args['model']}_output.png")
plt.show()
plt.close()