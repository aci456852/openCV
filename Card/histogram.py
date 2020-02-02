import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_demo(image):
    #  粗暴型直方图
    plt.hist(image.ravel(),256,[0,256])
    plt.show()

def image_hist(image):
    #  自绘直方图
    color = ('blue','green','red')
    for i, color in enumerate(color):  #  从容其中迭代
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()

def create_rgb_hist(image):
    #  绘制GRB直方图
    h, w, c = image.shape
    rgbHist = np.zeros([16*16*16, 1], np.float32)  #  int的话会报错
    bsize = 256/16
    #  降维
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int(b/bsize)*16*16 + np.int(g/bsize)*16 + np.int(r/bsize)
            rgbHist[np.int(index), 0] = rgbHist[np.int(index), 0] + 1
    return rgbHist

def hist_compare(image1, image2):
    #  直方图比较
    hist1 = create_rgb_hist(image1)
    hist2 = create_rgb_hist(image2)
    match1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)  #  巴氏距离 0-1
    match2 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)  #  相关性 0-1
    match3 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)  #  卡方 越大越相关
    print("巴氏距离:%s, 相关性:%s, 卡方:%s" % (match1, match2, match3))

def equalHist_demo(image):
    #  直方图均衡化 基于灰度图像
    #  调整对比度 是图像增强一个手段
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    cv2.imshow("equalHist_demo",dst)

def clahe_demo(image):
    #  局部直方图均衡化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #  clipLimit越大 对比度越大
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dst = clahe.apply(gray)
    cv2.imshow("clahe_demo",dst)

def hist2d_demo(image):
    #  HSV直方图
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #  32*32是显示的大小，可修改。而 [0, 180, 0, 256]是固定的，不可修改
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    plt.imshow(hist, interpolation='nearest')
    plt.title("2D Histogram")
    plt.show()

def back_projection_demo():
    #  直方图的反向投影
    #  sample在target中的部分呈白色，其他黑底
    sample = cv2.imread('tt.jpg')  #  采集的样本
    target = cv2.imread('tx.jpg')  #  待检测目标（原图）
    roi_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    cv2.imshow("sample", sample)
    cv2.imshow("target", target)
    roiHist = cv2.calcHist([roi_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([target_hsv], [0, 1], roiHist, [0, 180, 0, 256], 1)
    cv2.imshow("back_projection_demo", dst)

img = cv2.imread('tx.jpg')
img2 = cv2.imread('tx2.jpg')  #  作比较用
cv2.namedWindow("imshow",cv2.WINDOW_AUTOSIZE)
#  cv2.imshow('imshow', img)

hist2d_demo(img)
back_projection_demo()

cv2.waitKey(0)
cv2.destroyAllWindows()