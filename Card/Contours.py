import cv2


#  轮廓发现
#  基于图像边缘提取（二值图像）
#  利用梯度来避免阈值混乱

def contours_demo(image):
    dst = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("binary image", binary)
    #  findContours 发现轮廓
    contours, heriachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv2.drawContours(image, contours, i, (0, 0, 255), -1)  #  drawContours 绘制轮廓  -1是填充轮廓
        print(i)
    cv2.imshow("contours_demo", image)


img = cv2.imread('coin.jpg')
cv2.namedWindow("imshow",cv2.WINDOW_AUTOSIZE)
cv2.imshow('imshow', img)

contours_demo(img)

cv2.waitKey(0)
cv2.destroyAllWindows()