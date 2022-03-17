import matplotlib.pyplot as plt
import cv2
import glob


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


image_paths = glob.glob('./input/*.jpg')
print(f"Found {len(image_paths)} images...")
plt.figure(figsize=(16, 12))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profilface.xml")

img_with_edit = []
img_without_edit = []
img_incorrect_edit = []

plt.axis("off")
for i, image_path in enumerate(image_paths):
    orig_image = plt.imread(image_path)

    img = FaceRecog(orig_image, face_cascade, eye_cascade, smile_cascade, profile_face_cascade)

    plt.subplot(5, 10, 2 * i + 1)
    plt.imshow(img)
    plt.axis('off')

plt.show()
