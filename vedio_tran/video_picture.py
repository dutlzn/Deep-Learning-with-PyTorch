#抽取视频的帧 构成图像
import cv2 
import numpy as np 
import os 
import shutil 

#生成文件夹
def mkdir(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

mkdir('../../data/video_picture')
videoCapture = cv2.VideoCapture()
videoCapture.open('./v_YoYo_g07_c04.avi')
fps = videoCapture.get(cv2.CAP_PROP_FPS)#统计视频的帧率
frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) #统计视频的总帧数
#fps是帧率，意思是每一秒刷新图片的数量，frames是一整段视频中总的图片数量。
print("fps=",fps,"frames=",frames)

for i in range(int(frames)):
    ret,frame = videoCapture.read()
    cv2.imwrite("../../data/video_picture/v_YoYo_g07_c04.avi(%d).jpg"%i,frame)
