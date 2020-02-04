import cv2
import numpy as np
import matplotlib.pyplot as plt


def threshold_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #  图像二值化 - 全局
    #  最大类间方差法（OTSU）是找到自适应阈值的常用方法
    #  THRESH_BINARY 如果 src(x,y)>threshold,dst(x,y) = max_value; 否则,dst（x,y）=0;
    #  THRESH_OTSU 最大类间方差法（OTSU）是找到自适应阈值的常用方法
    #  THRESH_TRUNC 如果 src(x,y)>threshold,dst(x,y) = max_value; 否则dst(x,y) = src(x,y).
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    print("threshold value %s"%ret)
    cv2.imshow("binary", binary)

def local_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #  图像二值化 - 局部
    #  blocksize 必须是奇数 25
    #  dst-None 可以看作是去噪，比均值大10才为白色
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
    cv2.imshow("binary", binary)

def custom_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #  图像二值化 - 自定义 自己计算阈值代入
    h, w = gray.shape[:2]
    m = np.reshape(gray, [1, w+h])
    mean = m.sum() / (w*h)
    print("mean: ", mean)
    ret, binary = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary", binary)

def big_demo(image):
    # 对于大图片的二值化 需要分块处理
    cw = 256
    ch = 256
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            roi = gray[row:row+ch, col:col+cw]
            dev = np.std(roi)
            if dev < 15: #  空白图像过滤
                gray[row:row + ch, col:col + cw] = 255
            else:
                ret, dst = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                gray[row:row+ch, col:col+cw] = dst
    cv2.imwrite("result_binary.jpg", gray)


img = cv2.imread('tx.jpg')
cv2.namedWindow("imshow",cv2.WINDOW_AUTOSIZE)
cv2.imshow('imshow', img)

threshold_demo(img)
local_threshold(img)

cv2.waitKey(0)
cv2.destroyAllWindows()