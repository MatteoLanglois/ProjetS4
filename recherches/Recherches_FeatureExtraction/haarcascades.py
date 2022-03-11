import sys
import cv2

x=y=w=h=0

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# Create the haar cascade
noseCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread("projets4/dataset/test/incorrect_mask/incorrect_mask_1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Draw a rectangle around the faces
roi_gray = gray[y:y+h, x:x+w]
roi_color = image[y:y+h, x:x+w]
nose = noseCascade.detectMultiScale(roi_gray)

for (ex,ey,ew,eh) in nose:
    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.waitKey(0)