import cv2
import numpy as np

#  图像梯度
def sobel_demo(image):
    #  一阶导数 Soble算子
    #  Scharr是对Sobel的部分优化  cv2.Scharr
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    gradx = cv2.convertScaleAbs(grad_x)
    grady = cv2.convertScaleAbs(grad_y)
    cv2.imshow("gradient-x", gradx)
    cv2.imshow("gradient-y", grady)

    gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
    cv2.imshow("gradient", gradxy)

def lapalian_demo(image):
    #  二阶导数 拉普拉斯算子
    #   dst = cv2.Laplacian(image, cv2.CV_32F) 默认是 0 1 0  1 -4 1  0 1 0
    kernel = np.array([[1, 1, 1],[1, -8, 1],[1, 1, 1]])
    dst = cv2.filter2D(image, cv2.CV_32F, kernel=kernel)
    lpls = cv2.convertScaleAbs(dst)
    cv2.imshow("lapalian_demo", lpls)


img = cv2.imread('tx.jpg')
cv2.namedWindow("imshow",cv2.WINDOW_AUTOSIZE)
cv2.imshow('imshow', img)

sobel_demo(img)
lapalian_demo(img)

cv2.waitKey(0)
cv2.destroyAllWindows()