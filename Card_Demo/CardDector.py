import cv2
import numpy as np

#  [打开摄像头] 0 内置 1USB
def video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # 左右颠倒
        if not ret:
            break
        frame = imutils.resize(frame, width=720)
        cv2.imshow('', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q退出
            break


#  [处理图片] 灰度化->高斯模糊->二值化
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #  cv2.imshow('blur', blur)
    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h / 100)][int(img_w / 2)]
    thresh_level = bkg_level + 90  # 阈值 需要根据情况调节
    retval, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)
    #  cv2.imshow('thresh', thresh)
    return thresh


#  [寻找扑克牌] 边缘检测->按面积排序->根据面积大小和边界4判断是否为扑克牌，用cnt_is_card数组存储标记
def find_cards(thresh_image):

    #  面积的阈值根据摄像头离扑克牌的远近进行修改
    CARD_MAX_AREA = 1200000
    CARD_MIN_AREA = 25000

    cnts, hier = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)   # 根据轮廓的面积进行排序

    if len(cnts) == 0:
        return [], []

    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts), dtype=int)

    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i], True)
        approx = cv2.approxPolyDP(cnts_sort[i], 0.01 * peri, True)  # 直线近似化轮廓

        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA) and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card


# 矩形矫正 将倾斜的矩形矫正为水平/垂直的矩形
def adjust_image(image, pts, w, h):
    temp_rect = np.zeros((4, 2), dtype="float32")

    s = np.sum(pts, axis=2)  # 前2个数字相加 x+y
    tl = pts[np.argmin(s)]  # 左上角
    br = pts[np.argmax(s)]  # 右下角

    diff = np.diff(pts, axis=-1)  # 后一个元素减去前一个元素
    tr = pts[np.argmin(diff)]  # 右上角
    bl = pts[np.argmax(diff)]  # 左下角

    if w <= 0.8 * h:  # 竖直 顺时针一圈
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2 * h:  # 水平
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    if 0.8 * h < w < 1.2 * h:  # 倾斜
        if pts[1][0][1] <= pts[3][0][1]:
            temp_rect[0] = pts[1][0]
            temp_rect[1] = pts[0][0]
            temp_rect[2] = pts[3][0]
            temp_rect[3] = pts[2][0]

        if pts[1][0][1] > pts[3][0][1]:
            temp_rect[0] = pts[0][0]
            temp_rect[1] = pts[3][0]
            temp_rect[2] = pts[2][0]
            temp_rect[3] = pts[1][0]

    maxWidth = 200
    maxHeight = 300

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect, dst)  # 转换矩阵
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))  # 透视变换
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    return warp


img = cv2.imread('C:/Users/10846/PycharmProjects/Card_Demo/venv/Lib/example/test.jpg')
#  cv2.imshow('img', img)

thresh = preprocess_image(img)
#  cv2.imshow('thresh', thresh)


cnts_sort, cnt_is_card = find_cards(thresh)

#  红色描出图像外边框 绿色描出方形框
for i, contour in enumerate(cnts_sort):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    if cnt_is_card[i] == 1:
        cv2.drawContours(img, cnts_sort, i, (0, 0, 255), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.namedWindow("draw frame", 0)
cv2.resizeWindow("draw frame", 700, 500)
cv2.imshow("draw frame", img)


print(len(cnts_sort))
print(cnt_is_card)

#  输出找到的单张扑克牌
count = 0
for flag in cnt_is_card:
    if flag == 1:
        x, y, w, h = cv2.boundingRect(cnts_sort[count])
        out = adjust_image(img, cnts_sort[count], w, h)
        cv2.imshow("out:%s" % count, out)
        count += 1

cv2.waitKey(0)