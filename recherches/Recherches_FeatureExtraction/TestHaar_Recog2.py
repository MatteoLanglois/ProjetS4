import matplotlib.pyplot as plt
from imutils import face_utils
import imutils
import dlib
import cv2


def recog(img, detector, predictor):
    img_gray = [cv2.cvtColor(I, cv2.COLOR_BGR2GRAY) for I in img]
    img_rects = [detector(I, 1) for I in img]
    for (imageID, rects) in enumerate(img_rects):
        for (i, rect) in enumerate(rects):
            # déterminer les repères du visage for the face region, then
            # convertir le repère du visage (x, y) en un array NumPy
            shape = predictor(img_gray[imageID], rect)
            shape = face_utils.shape_to_np(shape)
            # convertir le rectangle de Dlib en un cadre de sélection de style OpenCV
            # dessiner le cadre de sélection
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(img[imageID], (x, y), (x + w, y + h), (0, 255, 0), 1)
            # boucle sur les coordonnées (x, y) pour les repères faciaux
            # et dessine-les sur l'image
            for (x, y) in shape:
                cv2.circle(img[imageID], (x, y), 1, (0, 0, 255), -1)
        # afficher l'image de sortie avec les détections de visage + repères de visage
        plt.subplot(1, 2, imageID + 1)
        cv2.imshow("Output", img[imageID])


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "./recherches/Recherches_FeatureExtraction/data/shape_predictor_68_face_landmarks.dat")

img_with = [cv2.imread(f'./dataset/train/with_mask/with_mask_{I}.jpg') for I in range(0, 2)]
img_without = [cv2.imread(f'./dataset/train/without_mask/without_mask_{I}.jpg') for I in range(0, 2)]
img_incorrect = [cv2.imread(f'./dataset/train/incorrect_mask/incorrect_mask_{I}.jpg') for I in [0, 3]]

img_with = [imutils.resize(I, width=500) for I in img_with]
img_without = [imutils.resize(I, width=500) for I in img_without]
img_incorrect = [imutils.resize(I, width=500) for I in img_incorrect]

recog(img_with, detector, predictor)
recog(img_without, detector, predictor)
recog(img_incorrect, detector, predictor)

plt.show()
