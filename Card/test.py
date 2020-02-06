import cv2
import numpy as np
import matplotlib.pyplot as plt




img = cv2.imread('line.jpg')
cv2.namedWindow("imshow",cv2.WINDOW_AUTOSIZE)
cv2.imshow('imshow', img)



cv2.waitKey(0)
cv2.destroyAllWindows()