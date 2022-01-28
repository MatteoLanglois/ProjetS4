from cv2 import *

img = cv2.imread('../images/Nana.jpg')

t_lower = 100
t_upper = 200
aperture_size = 3
L2Gradient = True

edge = cv2.Canny(img, t_lower, t_upper, apertureSize=aperture_size, L2gradient=L2Gradient)

img = cv2.resize(img, dsize=(600, 900), interpolation=cv2.INTER_CUBIC)
edge = cv2.resize(edge, dsize=(600, 900), interpolation=cv2.INTER_CUBIC)


cv2.imshow('original', img)
cv2.imshow('edge', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()