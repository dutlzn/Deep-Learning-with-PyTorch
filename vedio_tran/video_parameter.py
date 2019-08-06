import os
import cv2
import numpy as np

video_cap = cv2.VideoCapture('./v_YoYo_g07_c04.avi')

frame_count = 0
all_frames = []
while(True):
    ret, frame = video_cap.read()
    if ret is False:
        break
    all_frames.append(frame)
    frame_count = frame_count + 1

print(frame_count)
print(len(all_frames))
all = np.array(all_frames)
print(all.shape)
n, w, h, p = all.shape
# print(n, w, h, p)#元祖的解包
w = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print("width:", w)
h = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("height:", h)

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print(major_ver)
if int(major_ver)  < 3 :
    fps = video_cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

video_cap.release(); 

