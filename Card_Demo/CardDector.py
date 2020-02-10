import cv2
import numpy as np
import matplotlib.pyplot as plt
import myutils
import imutils
import argparse


#  输出
def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#  读入文件
img = cv2.imread('C:/Users/10846/PycharmProjects/Card_Demo/venv/Lib/images/all.jpg')
#  cv2.namedWindow("example", cv2.WINDOW_AUTOSIZE)

#  灰度图像
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#  二值图像
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

#  轮廓检测
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)  # 用红色画出轮廓 -1表示画出所有轮廓 2表示线的粗细
#  show('model', img)
print(np.array(refCnts).shape)  #  打印出有几个轮廓

refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# 遍历每一个轮廓
for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    digits[i] = roi  # 对应

# 初始化卷积核 根据纸牌大小指定
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

#读取输入图像，预处理
image = cv2.imread('C:/Users/10846/PycharmProjects/Card_Demo/venv/Lib/example/Hearts3.jpg')
#  show('target', image)
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#突出更明亮的区域: 开运算(先腐蚀，再膨胀，可清除一些小东西，放大局部低亮度的区域)
tophat = cv2.morphologyEx(gray, cv2.MORPH_OPEN, rectKernel)
show('tophat', tophat)

