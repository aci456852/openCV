import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/10846/PycharmProjects/Card_Demo/venv/Lib/example/Hearts8.JPG')
cv2.namedWindow("example", cv2.WINDOW_AUTOSIZE)
cv2.imshow('example', img)

#  灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)

#均值滤波  去除噪声
kernel = np.ones((3, 3), np.float32) / 9
gray = cv2.filter2D(gray, -1, kernel)
cv2.imshow('gray2', gray)

blur = cv2.GaussianBlur(gray, (5, 5), 0, 0, cv2.BORDER_DEFAULT)  #  高斯平滑
ret, binary=cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)  #  二值化处理


cv2.waitKey(0)
cv2.destroyAllWindows()