import cv2
import numpy as np

def fill_color(image):
    # 填充图像
    copyImg=image.copy()
    h,w=image.shape[:2]
    mask=np.zeros([h+2,w+2],np.uint8)
    #泛洪填充 像素值范围
    cv2.floodFill(copyImg,mask,(30,30),(0,255,255),(100,100,100),(50,50,50),cv2.FLOODFILL_FIXED_RANGE)
    cv2.imshow("fill_color demo",copyImg)

def fill_binary(image):
    # 填充图像 二进制
    image=np.zeros([400,400,3],np.uint8) #原图像
    image[100:300,100:300,:]=255
    cv2.imshow("fill_binary demo",image)
    mask=np.ones([402,402,1],np.uint8)
    mask[101:301,101:301]=0
    cv2.floodFill(image,mask,(200,200),(0,0,255),cv2.FLOODFILL_MASK_ONLY)#填充
    cv2.imshow("fill_binary demo2", image)

img = cv2.imread('tx.jpg')
cv2.namedWindow("imshow",cv2.WINDOW_AUTOSIZE)
cv2.imshow('imshow', img)
fill_color(img)
fill_binary(img)

'''
#ROI操作 常用于合并图像
face=img[130:330,200:400] #截取想要的图像
gray=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY) #灰度图像
backface=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
img[130:330,200:400]=backface #替换回去
cv2.imshow("face",img)
'''

cv2.waitKey(0)
cv2.destroyAllWindows()