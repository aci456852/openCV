import cv2
import numpy as np


def blur_demo(image):
    # 均值模糊 卷积原理
    dst = cv2.blur(image,(5,5))
    cv2.imshow("blur",dst)

def median_blur_demo(image):
    # 中值模糊 适用于椒盐噪声
    dst = cv2.medianBlur(image,5)
    cv2.imshow("median_blur",dst)

def custom_blur_demo(image):
    # 自定义模糊
    # kernel=np.ones([5,5],np.float32)/25 模糊
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32) #锐化
    dst = cv2.filter2D(image,-1,kernel=kernel)
    cv2.imshow("custom_blur", dst)

def clamp(pv):
    if pv>255:
        return 255
    if pv<0:
        return 0
    else:
        return pv

def gaussian_noise(image):
    # 高斯 加噪声
    h, w, c = image.shape
    for row in range(0,h,1):
        for col in range(w):
            s = np.random.normal(0,20,3)
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    cv2.imshow("noise image",image)

img = cv2.imread('tx.jpg')
cv2.namedWindow("imshow",cv2.WINDOW_AUTOSIZE)
cv2.imshow('imshow', img)

gaussian_noise(img)
dst=cv2.GaussianBlur(img,(5,5),0) #高斯模糊
cv2.imshow("GaussianBlur image",dst)

cv2.waitKey(0)
cv2.destroyAllWindows()