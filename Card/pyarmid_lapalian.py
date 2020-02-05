import cv2


def pyarmid_demo(image):
    #  高斯金字塔
    #  reduce=高斯模糊+降采样（pyrDown）
    level = 3
    temp = image.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv2.pyrDown(temp)
        pyramid_images.append(dst)
        cv2.imshow("pyarmid_demo_"+str(i),dst)
        temp = dst.copy()
    return pyramid_images

def lapalian_demo(image):
    #  拉普拉斯金字塔 原图必须是2的n次方
    #  expand=扩大+卷积（pyrUp）
    pyramid_images = pyarmid_demo(image)
    level = len(pyramid_images)
    for i in range(level-1, -1, -1): #  递减到-1
        if (i-1) <0 : #  最后一层 对原图进行处理
            expand = cv2.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv2.subtract(image, expand)
            cv2.imshow("lapalian_demo_" + str(i), lpls)
        else:
            expand = cv2.pyrUp(pyramid_images[i], dstsize=pyramid_images[i-1].shape[:2])
            lpls = cv2.subtract(pyramid_images[i-1], expand)
            cv2.imshow("lapalian_demo_"+str(i), lpls)


img = cv2.imread('tx.jpg')
cv2.namedWindow("imshow",cv2.WINDOW_AUTOSIZE)
#cv2.imshow('imshow', img)

pyarmid_demo(img)
lapalian_demo(img)

cv2.waitKey(0)
cv2.destroyAllWindows()