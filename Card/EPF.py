import cv2
import numpy as np

#边缘保留滤波 EPF

def bi_demo(image):
    #高斯双边模糊 保留轮廓 相当于美颜相机的滤镜
    dst=cv2.bilateralFilter(image,0,100,15)
    cv2.imshow("bi demo",dst)

def shift_demo(image):
    #均值迁移 类似油画效果
    dst=cv2.pyrMeanShiftFiltering(image,10,50)
    cv2.imshow("shift demo",dst)

img = cv2.imread('tx.jpg')
cv2.namedWindow("imshow",cv2.WINDOW_AUTOSIZE)
cv2.imshow('imshow', img)
bi_demo(img)
shift_demo(img)

cv2.waitKey(0)
cv2.destroyAllWindows()