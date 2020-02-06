import cv2
import numpy as np
import matplotlib.pyplot as plt

#  Canny 边缘检测
#  1.高斯模糊 2.灰度转换 3.计算梯度 4.非最大信号抑制 5.高低阈值输出二值图像
def edge_demo(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    xgrad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    edge_output = cv2.Canny(xgrad, ygrad, 50, 150)
    cv2.imshow("Canny Edge", edge_output)

    dst = cv2.bitwise_and(image, image, mask=edge_output)
    cv2.imshow("Color Edge", dst)

img = cv2.imread('tx.jpg')
cv2.namedWindow("imshow",cv2.WINDOW_AUTOSIZE)
cv2.imshow('imshow', img)

edge_demo(img)

cv2.waitKey(0)
cv2.destroyAllWindows()