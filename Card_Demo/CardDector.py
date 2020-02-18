import cv2
import numpy as np
import imutils
import time

# 二值化阈值
BKG_THRESH = 60
CARD_THRESH = 30

# 根据摄像头距离改 是截取的
CORNER_WIDTH = 32
CORNER_HEIGHT = 84

# 数字图片大小
RANK_WIDTH = 70
RANK_HEIGHT = 125

# 花色图片大小
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

# 差分的最大值
RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700

CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000

font = cv2.FONT_HERSHEY_SIMPLEX

class Query_card:
    def __init__(self):
        self.contour = []                   # 轮廓
        self.width, self.height = 0, 0      # 宽度和高度
        self.corner_pts = []                # 多边形拟合后新的轮廓坐标
        self.center = []                    # 中心点  -没用到
        self.warp = []                      # 存放经过adjust_image函数处理得到的矩阵
        self.rank_img = []                  # 数字
        self.suit_img = []                  # 花色
        self.best_rank_match = "Unknown"    # 数字匹配的最佳结果
        self.best_suit_match = "Unknown"    # 花色匹配的最佳结果
        self.rank_diff = 0                  # 数字差距数值
        self.suit_diff = 0                  # 花色差距数值

# 数字模版类
class Train_ranks:
    def __init__(self):
        self.img = []
        self.name = "Placeholder"

# 加载数字模版图片
def load_ranks(filepath):
    train_ranks = []
    i = 0
    for Rank in ['Ace', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven','Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King']:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.jpg'
        train_ranks[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1
    return train_ranks


# 花色模版类
class Train_suits:
    def __init__(self):
        self.img = []
        self.name = "Placeholder"

# 加载花色模版图片
def load_suits(filepath):
    train_suits = []
    i = 0
    for Suit in ['Spades', 'Diamonds', 'Clubs', 'Hearts']:
        train_suits.append(Train_suits())
        train_suits[i].name = Suit
        filename = Suit + '.jpg'
        train_suits[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1
    return train_suits


#  [处理图片] 灰度化->高斯模糊->二值化
def preprocess_card(image):
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
def find_card(thresh_image):

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


# [矩形矫正] 将倾斜的矩形矫正为水平/垂直的矩形
def adjust_card(image, pts, w, h):
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

#  [得到数字和花色] 多边形拟合->将新的矩形图像放大四倍->切割数字和花色
def details_card(contour, image):
    qCard = Query_card()
    qCard.contour = contour

    peri = cv2.arcLength(contour, True)  # 得到轮廓周长
    approx = cv2.approxPolyDP(contour, 0.01 * peri, True)  # 多边形拟合
    pts = np.float32(approx)
    qCard.corner_pts = pts

    x, y, w, h = cv2.boundingRect(contour)  # 矩形边框：得到轮廓的左上角坐标，宽度和高度
    qCard.width, qCard.height = w, h
    average = np.sum(pts, axis=0) / len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    qCard.warp = adjust_card(image, pts, w, h)
    Qcorner = qCard.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    Qcorner_zoom = cv2.resize(Qcorner, (0, 0), fx=4, fy=4)  # 将新的矩形图像放大四倍

    white_level = Qcorner_zoom[15, int((CORNER_WIDTH * 4) / 2)]
    thresh_level = white_level - CARD_THRESH
    if thresh_level <= 0:
        thresh_level = 1

    retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2. THRESH_BINARY_INV)  # 对放大的图像二值化处理

    # 切割左上角位置，分别得到数值位置图像矩阵，花色位置图像矩阵
    Qrank = query_thresh[20:185, 0:128]
    Qsuit = query_thresh[186:336, 0:128]

    # 处理数值
    Qrank_cnts, hier = cv2.findContours(Qrank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea, reverse=True)
    if len(Qrank_cnts) != 0:
        x1, y1, w1, h1 = cv2.boundingRect(Qrank_cnts[0])
        Qrank_roi = Qrank[y1:y1 + h1, x1:x1 + w1]
        Qrank_sized = cv2.resize(Qrank_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
        qCard.rank_img = Qrank_sized

    # 处理花色
    Qsuit_cnts, hier = cv2.findContours(Qsuit, cv2.RETR_TREE ,cv2.CHAIN_APPROX_SIMPLE)
    Qsuit_cnts = sorted(Qsuit_cnts, key=cv2.contourArea, reverse=True)
    if len(Qsuit_cnts) != 0:
        x2, y2, w2, h2 = cv2.boundingRect(Qsuit_cnts[0])
        Qsuit_roi = Qsuit[y2:y2 + h2, x2:x2 + w2]
        Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        qCard.suit_img = Qsuit_sized

    return qCard

# [对比花色] 将得到的ROI区域和模版进行对比
def match_card(qCard, train_ranks, train_suits):

    best_rank_match_diff = 10000
    best_suit_match_diff = 10000
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"

    best_rank_name = ''
    best_suit_name = ''

    if (len(qCard.rank_img) != 0) and (len(qCard.suit_img) != 0):

        for Trank in train_ranks:
                diff_img = cv2.absdiff(qCard.rank_img, Trank.img)  # 返回的结果是他们的差矩阵
                rank_diff = int(np.sum(diff_img) / 255)  # 矩阵的全部值求和，然后标准化
                if rank_diff < best_rank_match_diff:
                    best_rank_match_diff = rank_diff
                    best_rank_name = Trank.name

        for Tsuit in train_suits:
                diff_img = cv2.absdiff(qCard.suit_img, Tsuit.img)
                suit_diff = int(np.sum(diff_img) / 255)
                if suit_diff < best_suit_match_diff:
                    best_suit_match_diff = suit_diff
                    best_suit_name = Tsuit.name

    if best_rank_match_diff < RANK_DIFF_MAX:
        best_rank_match_name = best_rank_name

    if best_suit_match_diff < SUIT_DIFF_MAX:
        best_suit_match_name = best_suit_name

    # 返回最佳匹配数数字和花色，及相应的差值
    return best_rank_match_name, best_suit_match_name, best_rank_match_diff, best_suit_match_diff


'''
单张图测试 
img = cv2.imread('C:/Users/10846/PycharmProjects/Card_Demo/venv/Lib/example/test.jpg')
cv2.imshow('img', img)
thresh = preprocess_card(img)
cv2.imshow('thresh', thresh)
cnts_sort, cnt_is_card = find_card(thresh)
'''

'''
# 红色描出图像外边框 绿色描出方形框
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

# 输出找到的单张扑克牌
count = 0
for flag in cnt_is_card:
    if flag == 1:
        x, y, w, h = cv2.boundingRect(cnts_sort[count])
        out = adjust_card(img, cnts_sort[count], w, h)
        # cv2.imshow("out:%s" % count, out)
        count += 1

# 输出数字和花色
count = 0
for flag in cnt_is_card:
    if flag == 1:
        out = details_card(cnts_sort[count], img)
        # cv2.imshow('rank_img %s' % count, out.rank_img)
        # cv2.imshow('suit_img %s' % count, out.suit_img)
        count += 1

# 输出匹配结果并在原图上绘制结果
train_ranks = load_ranks('C:/Users/10846/PycharmProjects/Card_Demo/venv/Lib/images/')
train_suits = load_suits('C:/Users/10846/PycharmProjects/Card_Demo/venv/Lib/images/')
count = 0
for flag in cnt_is_card:
    if flag == 1:
        out = details_card(cnts_sort[count], img)
        out.best_rank_match, out.best_suit_match, out.rank_diff, out.suit_diff = match_card(out, train_ranks,train_suits)
        print(out.best_rank_match, out.best_suit_match, out.rank_diff, out.suit_diff)

        x = out.center[0]
        y = out.center[1]
        # cv2.circle(img, (x, y), 5, (255, 0, 0), -1) 中心点

        cv2.putText(img, (out.best_rank_match + ' of ' + out.best_suit_match), (x - 210, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

        count += 1

cv2.namedWindow("result img", 0)
cv2.resizeWindow("result img", 700, 500)
cv2.imshow('result img', img)

'''


train_ranks = load_ranks('C:/Users/10846/PycharmProjects/Card_Demo/venv/Lib/images/')
train_suits = load_suits('C:/Users/10846/PycharmProjects/Card_Demo/venv/Lib/images/')
frame_rate_calc = 1
freq = cv2.getTickFrequency()

cap = cv2.VideoCapture(1) #0为笔记本摄像头 1为USB外接摄像头
time.sleep(1)

cam_quit = 0

while True:
    ret, image = cap.read()
    # image = cv2.flip(image, 1)  # 左右颠倒

    if not ret:
        break

    t1 = cv2.getTickCount()
    pre_proc = preprocess_card(image)
    cnts_sort, cnt_is_card = find_card(pre_proc)

    if len(cnts_sort) != 0:
        cards = []
        count = 0
        for i in range(len(cnts_sort)):
            if cnt_is_card[i] == 1:
                cards.append(details_card(cnts_sort[i], image))
                cards[count].best_rank_match, cards[count].best_suit_match, cards[count].rank_diff, cards[count].suit_diff = match_card(cards[count], train_ranks, train_suits)
                x = cards[count].center[0]
                y = cards[count].center[1]
                cv2.putText(image, (cards[count].best_rank_match + ' . ' + cards[count].best_suit_match), (x - 100, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                count = count + 1

        if len(cards) != 0:
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(image, temp_cnts, -1, (255, 0, 0), 2)

    # cv2.putText(image, "FPS: " + str(int(frame_rate_calc)), (10, 26), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("", image)

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
