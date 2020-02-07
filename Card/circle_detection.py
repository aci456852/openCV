import cv2
import numpy as np
import matplotlib.pyplot as plt

#  圆检测 霍夫圆变换（平面坐标与极坐标转换 极坐标中最亮的一点表示圆心）
#  1.边缘检测，发现可能的圆心
#  2.从候选圆心开始计算最佳半径大小
def circle_detection(image):
    dst = cv2.pyrMeanShiftFiltering(image, 10, 120)  #  消除噪声
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    #  100是最小距离 在这个距离之内的圆会被合并
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 100, param1=70, param2=35, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)
    cv2.imshow("circles",image)


#  数据调不好.. 凑合一下...

img = cv2.imread('coin.jpg')
cv2.namedWindow("imshow",cv2.WINDOW_AUTOSIZE)
cv2.imshow('imshow', img)

circle_detection(img)

cv2.waitKey(0)
cv2.destroyAllWindows()