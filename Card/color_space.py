import cv2
import numpy as np


def color_space(image):
    #RGB转换到不同的色彩空间
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",gray)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv",hsv)
    yuv=cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
    cv2.imshow("yuv",yuv)
    Ycrcb=cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
    cv2.imshow("Ycrcb",Ycrcb)

def extrace_object():
    cap = cv2.VideoCapture("C:/Users/10846/PycharmProjects/Card/mc.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        # 过滤颜色 想要的颜色白 其他黑
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        lower_hsv=np.array([0,0,0]) # 黑色
        upper_hsv=np.array([180,255,46])
        mask=cv2.inRange(hsv,lowerb=lower_hsv,upperb=upper_hsv)
        cv2.imshow("video", frame)
        cv2.imshow("mask", mask)

        #过滤颜色 只剩想要的颜色
        dst=cv2.bitwise_and(frame,frame,mask=mask)
        cv2.imshow("dst", dst)
        c = cv2.waitKey(1)
        if c == 27:
            break

def contrast_brightness(image,c,b):
    # 改变图片的对比度和亮度
    h,w,ch=image.shape
    blank=np.zeros([h,w,ch],image.dtype) # 创建一个与原图一样大的空白图片
    dst=cv2.addWeighted(image,c,blank,1-c,b)
    cv2.imshow("contrast_brightness demo",dst)

img = cv2.imread('tx.jpg')
cv2.namedWindow("imshow",cv2.WINDOW_AUTOSIZE)
cv2.imshow('imshow', img)

contrast_brightness(img,1.2,10) # 改变图片的对比度和亮度

# 通道分离
b,g,r=cv2.split(img)
cv2.imshow("blue",b)
cv2.imshow("green",g)
cv2.imshow("red",r)
img[:,:,2]=0
cv2.imshow("change img",img)
img=cv2.merge([b,g,r])
cv2.imshow("change back img",img)

# extrace_object()
cv2.waitKey(0)

cv2.destroyAllWindows()