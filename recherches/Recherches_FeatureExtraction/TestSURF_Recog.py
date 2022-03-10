import cv2
import numpy as np
import matplotlib.pyplot as plt

img_with = [cv2.imread(f'ProjetS4/dataset/train/with_mask/with_mask_{I}.jpg') for I in range(0, 2)]
img_without = [cv2.imread(f'ProjetS4/dataset/train/without_mask/without_mask_{I}.jpg') for I in range(0, 2)]
img_incorrect = [cv2.imread(f'ProjetS4/dataset/train/incorrect_mask/incorrect_mask_{I}.jpg') for I in [0, 3]]

