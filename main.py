import cv2

from func import main_detect, main_class, GUI_init


# 载入测试视频
vc = cv2.VideoCapture('video/dxandcar.mp4')
cap = cv2.VideoCapture('video/dxha.mp4')

if __name__ == '__main__':
    GUI_init() # 初始化GUI界面
    while 1:
        main_detect(cap)
        main_class(vc)
