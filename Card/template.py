import cv2
import numpy as np
import matplotlib.pyplot as plt


def template_demo():
    tpl = cv2.imread('tt.jpg')
    target = cv2.imread('tx.jpg')
    #  模板匹配 ：1平方差（最好的匹配值为0；匹配越差，匹配值越大）
    #  2乘法 （数值越大表明匹配程度越好）
    #  3相关系数（1表示完美的匹配；-1表示最差的匹配）
    #  NORMED 归一化
    methods = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]
    th, tw = tpl.shape[:2]
    for md in methods:
        print(md)
        result = cv2.matchTemplate(target, tpl, md)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if md == cv2.TM_SQDIFF_NORMED:
            tl = min_loc
        else:
            tl = max_loc
        br = (tl[0] + tw, tl[1] + th)
        cv2.rectangle(target, tl, br, (0, 0, 255), 2)
        cv2.imshow("mathch"+np.str(md), target)

img = cv2.imread('tx.jpg')
cv2.namedWindow("imshow",cv2.WINDOW_AUTOSIZE)
cv2.imshow('imshow', img)

template_demo()

cv2.waitKey(0)
cv2.destroyAllWindows()