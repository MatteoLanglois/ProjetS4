import matplotlib.pyplot as plt
from imutils import face_utils
import imutils
import dlib
import cv2


def recog(img, detector, predictor):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rects = detector(img, 1)
    for (i, rect) in enumerate(img_rects):
        # déterminer les repères du visage for the face region, then
        # convertir le repère du visage (x, y) en un array NumPy
        shape = predictor(img_gray, rect)
        shape = face_utils.shape_to_np(shape)
        # convertir le rectangle de Dlib en un cadre de sélection de style OpenCV
        # dessiner le cadre de sélection
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        nose = shape[27:36]
        mouth = shape[48:60]
        mask = shape[1:16]
        # convertir le rectangle de Dlib en un cadre de sélection de style OpenCV
        # dessiner le cadre de sélection
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # boucle sur les coordonnées (x, y) pour les repères faciaux
        # et dessine-les sur l'image
        for (x, y) in left_eye:
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        for (x, y) in right_eye:
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        for (x, y) in nose:
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
        for (x, y) in mouth:
            cv2.circle(img, (x, y), 3, (255, 255, 0), -1)
        for (x, y) in mask:
            cv2.circle(img, (x, y), 3, (0, 255, 255), -1)
    # afficher l'image de sortie avec les détections de visage + repères de visage
    return img


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "./recherches/Recherches_FeatureExtraction/data/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = recog(img, detector, predictor)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
