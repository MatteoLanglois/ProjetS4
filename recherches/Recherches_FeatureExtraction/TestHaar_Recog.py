import matplotlib.pyplot as plt
import cv2


def FaceRecog(img, face_cascade, eye_cascade, smile_cascade, profile_face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        smiles = smile_cascade.detectMultiScale(roi_gray)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
    return img


img_with = [cv2.imread(f'ProjetS4/dataset/train/with_mask/with_mask_{I}.jpg') for I in range(0, 2)]
img_without = [cv2.imread(f'ProjetS4/dataset/train/without_mask/without_mask_{I}.jpg') for I in range(0, 2)]
img_incorrect = [cv2.imread(f'ProjetS4/dataset/train/incorrect_mask/incorrect_mask_{I}.jpg') for I in [0, 3]]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profilface.xml")

img_with_edit = []
img_without_edit = []
img_incorrect_edit = []

plt.axis("off")
for I in range(len(img_with)):
    img_with_edit = FaceRecog(img_with[I], face_cascade, eye_cascade, smile_cascade, profile_face_cascade)
    img_without_edit = FaceRecog(img_without[I], face_cascade, eye_cascade, smile_cascade, profile_face_cascade)
    img_incorrect_edit = FaceRecog(img_incorrect[I], face_cascade, eye_cascade, smile_cascade, profile_face_cascade)

    plt.subplot(3, 3, I + 1)
    plt.imshow(cv2.cvtColor(img_with[I], cv2.COLOR_RGB2BGR))
    plt.subplot(3, 3, I + 4)
    plt.imshow(cv2.cvtColor(img_without[I], cv2.COLOR_RGB2BGR))
    plt.subplot(3, 3, I + 7)
    plt.imshow(cv2.cvtColor(img_incorrect[I], cv2.COLOR_RGB2BGR))

plt.show()
