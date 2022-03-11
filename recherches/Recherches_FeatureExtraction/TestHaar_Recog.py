import matplotlib.pyplot as plt
import cv2

img_with = [cv2.imread(f'ProjetS4/dataset/train/with_mask/with_mask_{I}.jpg') for I in range(0, 2)]
img_without = [cv2.imread(f'ProjetS4/dataset/train/without_mask/without_mask_{I}.jpg') for I in range(0, 2)]
img_incorrect = [cv2.imread(f'ProjetS4/dataset/train/incorrect_mask/incorrect_mask_{I}.jpg') for I in [0, 3]]
harr_cascade = cv2.CascadeClassifier("ProjetS4/recherches/Recherches_FeatureExtraction/harr_face_detect_classifier.xml")

img_with_edit = []
img_without_edit = []
img_incorrect_edit = []

for I in range(len(img_with)):
    img_with_edit.append(cv2.resize(cv2.cvtColor(img_with[I], cv2.COLOR_BGR2GRAY), (600, 900), interpolation=cv2.INTER_CUBIC))
    img_without_edit.append(cv2.resize(cv2.cvtColor(img_without[I], cv2.COLOR_BGR2GRAY), (600, 900), interpolation=cv2.INTER_CUBIC))
    img_incorrect_edit.append(cv2.resize(cv2.cvtColor(img_incorrect[I], cv2.COLOR_BGR2GRAY), (600, 900), interpolation=cv2.INTER_CUBIC))

    face_cords_with = harr_cascade.detectMultiScale(img_with_edit[I], scaleFactor=1.1, minNeighbors=1)
    face_cords_without = harr_cascade.detectMultiScale(img_without_edit[I], scaleFactor=1.1, minNeighbors=1)
    face_cords_incorrect = harr_cascade.detectMultiScale(img_incorrect_edit[I], scaleFactor=1.1, minNeighbors=1)

    for x, y, w, h in face_cords_with:
        cv2.rectangle(img_with[I], (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

    for x, y, w, h in face_cords_without:
        cv2.rectangle(img_without[I], (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

    for x, y, w, h in face_cords_incorrect:
        cv2.rectangle(img_incorrect[I], (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

    plt.subplot(3, 3, I + 1)
    plt.imshow(cv2.cvtColor(img_with[I], cv2.COLOR_RGB2BGR))
    plt.subplot(3, 3, I + 4)
    plt.imshow(cv2.cvtColor(img_without[I], cv2.COLOR_RGB2BGR))
    plt.subplot(3, 3, I + 7)
    plt.imshow(cv2.cvtColor(img_incorrect[I], cv2.COLOR_RGB2BGR))

plt.show()
