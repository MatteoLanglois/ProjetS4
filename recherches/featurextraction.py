from mtcnn import MTCNN
import cv2

import os
cascPath = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(60, 60),
                                     flags=cv2.CASCADE_SCALE_IMAGE)



"""detector = MTCNN()
# Load a videopip TensorFlow
video_capture = cv2.VideoCapture(0)

while (True):
    ret, frame = video_capture.read()
    frame = cv2.resize(frame, (600, 400))
    boxes = detector.detect_faces(frame)
    if boxes:

        box = boxes[0]['box']
        conf = boxes[0]['confidence']
        x, y, w, h = box[0], box[1], box[2], box[3]

        if conf > 0.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()"""