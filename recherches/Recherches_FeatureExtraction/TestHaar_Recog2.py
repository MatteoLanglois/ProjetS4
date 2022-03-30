import matplotlib.pyplot as plt
from imutils import face_utils
import dlib
import cv2
import glob

def recog(img, detector, predictor):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rects = detector(img, 1)
    for (i, rect) in enumerate(img_rects):
        # déterminer les repères du visage for the face region, then
        # convertir le repère du visage (x, y) en un array NumPy
        shape = predictor(img_gray, rect)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        nose = shape[27:36]
        mouth = shape[48:60]
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
    # afficher l'image de sortie avec les détections de visage + repères de visage
    return img


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "./recherches/Recherches_FeatureExtraction/data/shape_predictor_68_face_landmarks.dat")

image_paths = glob.glob('./input/*.jpg')
print(f"Found {len(image_paths)} images...")
plt.figure(figsize=(16, 12))

plt.axis("off")
for i, image_path in enumerate(image_paths):
    orig_image = plt.imread(image_path)

    img = recog(orig_image, detector, predictor)

    plt.subplot(3, 5, i + 1)
    plt.imshow(img)
    plt.axis('off')

plt.show()
