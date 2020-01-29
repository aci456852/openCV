import cv2
import numpy as np

def get_image_info(image):
    print(type(image)) # 类型
    print(image.shape) # 高 宽 通道数
    print(image.size)  # 所占大小
    print(image.dtype) # 所占字节
    pixel_data=np.array(image)
    print(pixel_data)

def video_demo():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # 左右颠倒
        cv2.imshow("video", frame)
        c = cv2.waitKey(1)  # 参数是1，表示延时1ms切换到下一帧图像，参数过大如cv2.waitKey(1000)，会因为延时过久而卡顿感觉到卡顿。
                            # 参数为0，如cv2.waitKey(0)只显示当前帧图像，相当于视频暂停。
        if c==27:
            break
    cap.release()  # 释放视频
    cv2.destroyAllWindows()  # 关闭所有图像窗口

img = cv2.imread('tx.jpg')
cv2.namedWindow("imshow",cv2.WINDOW_AUTOSIZE)
cv2.imshow('imshow', img)
get_image_info(img)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 灰度图像
cv2.imwrite("tx2.jpg",gray)
cv2.waitKey(0)

cv2.destroyAllWindows()