import cv2
import numpy as np


def access_pixels(image):
    print(image.shape)
    height=image.shape[0]
    width=image.shape[1]
    channels=image.shape[2]
    print("width:%s height:%s channels:%s"%(width,height,channels))
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv=image[row,col,c]
                image[row,col,c]=255-pv
    cv2.imshow("pixels_demo",image)

def create_image():
    # 1 初始化图像
    img=np.zeros([400,400,3],np.uint8)
    img[:,:,0]=np.ones([400,400])*255
    # 2
    img=np.ones([400,400,3],np.uint8)
    img=img*255
    cv2.imshow("new image",img)

img = cv2.imread('tx.jpg')
cv2.namedWindow("imshow",cv2.WINDOW_AUTOSIZE)
cv2.imshow('imshow', img)
t1=cv2.getTickCount()
create_image()
t2=cv2.getTickCount()
time=(t2-t1)/cv2.getTickFrequency()
print("time:%s ms"%(time*1000))
cv2.waitKey(0)

cv2.destroyAllWindows()