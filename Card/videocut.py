import cv2
cap=cv2.VideoCapture("mc.mp4") #获取视频
isOpened=cap.isOpened()
fps=cap.get(cv2.CAP_PROP_FPS)#帧率
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
i=0
while(isOpened):
    if i==10:
        break
    else:
        i=i+1
    (flag,frame)=cap.read() #读取
    fileName='image'+str(i)+'.jpg'
    if flag==True:
        cv2.imwrite(fileName,frame,[cv2.IMWRITE_JPEG_CHROMA_QUALITY,100])