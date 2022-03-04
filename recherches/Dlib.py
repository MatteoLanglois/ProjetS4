import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = './dataset/'
offset = 10

file_name = input("Enter the name of the person :  ")
while True:
    ret, frame = cap.read()

    if (ret == False):
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    # print(faces)
    faces = sorted(faces, key=lambda f: f[2] * f[3])

    # pick the last face (largest)
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # extract main face
    face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
    face_section = cv2.resize(face_section, (100, 100))

    skip += 1
    if (skip % 10 == 0):
        face_data.append(face_section)
        print(len(face_data))

    cv2.imshow("Cropped", face_section)
    cv2.imshow("VIDEO FRAME", frame)

    keypressed = cv2.waitKey(1) & 0xFF
    if (keypressed == ord('q')):
        break

# convert our face list array into a numpy array
face_data = np.array(face_data)
print(face_data.shape)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

# save this data into file system
np.save(dataset_path + file_name + '.npy', face_data)
print("data successfully saved at " + dataset_path + file_name + '.npy')

import cv2
import numpy as np
import os


########## KNN CODE ############
def distance(v1, v2):
    # Eucledian
    return np.sqrt(((v1 - v2) ** 2).sum())


def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]


################################


# Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './data/'

face_data = []
labels = []

class_id = 0  # Labels for the given file
names = {}  # Mapping btw id - name# Data Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
    # Create a mapping btw class_id and name
        names[class_id] = fx[:-4]
        print('Loaded'+fx)
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)