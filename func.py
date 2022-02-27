import dlib
import joblib
import numpy as np
import copy
import pandas as pd
import pygame
from imutils import face_utils
from scipy.spatial import distance
from tkinter import *
from PIL import Image, ImageTk
import tkinter.ttk
import numpy
from PIL import Image, ImageDraw, ImageFont
import math
import cv2
import torch
from modules.pose import Pose, track_poses
from val import normalize, pad_width
from modules.keypoints import extract_keypoints, group_keypoints
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state

# 均值填充
avg = [108.3920476, 28.25560194, 23.83509614, 59.88074356, 38.39867032, 58.68210028, 169.0982822, 150.9596774,
       59.14090161, 107.8679241, 97.66179061, 229.0530985, 190.5525496, 109.007621, 207.7391332, 154.0163144,
       108.5794621, 2.335975213, -2.26903988, 1.427404258, -3.187685543, -1.970117366, -3.31392059, -4.290557895,
       -2.154595849, -4.042274581, -1.943534425, 2.783517288, -0.181217392, -0.706798676, -1.652686336, -0.950017573,
       1.550229334, 1.209989161]

switch = 0
yawn = False
yawn_flag = 0
eye_close = False
eye_flag = 0
flag = 0
t = 0

# 加载OpenPose模型
net = PoseEstimationWithMobileNet()  # 加载网络结构
checkpoint = torch.load('models/checkpoint_iter_370000.pth', map_location='cpu')  # 加载模型参数
load_state(net, checkpoint)  # 拼接结构与参数

thresh_eye = 0.17  # 眼睛宽高比阈值
thresh_mouth = 0.85  # 嘴巴宽高比阈值
frame_check = 25  # 超时警告（单位：帧）
detect = dlib.get_frontal_face_detector()  # 获取面部
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # 面部68个特征点数据集

# 获取眼睛特征点序号
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]  # 42~47
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]  # 36~41
# 获取嘴巴特征点序号
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]  # 48~67
pygame.mixer.init() # 语音模块初始化
# 载入分类模型
etc = joblib.load(
    'models/RandomForestClassifier_model.pkl')

datas = pd.DataFrame(
    columns=['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15', 'd16','d17',
             'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16','a17'])

# 中文显示函数
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
# 计算眼睛宽高比
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ratio = (A + B) / (2.0 * C)
    return ratio
# 计算嘴巴宽高比
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    ratio = (A + B) / (2.0 * C)
    return ratio
# 骨架缩放函数
def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1 / 256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)  # 标准化图片

    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()

    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)  # 神经网络输出

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad
# 获取骨架
def run_demo(net, img, height_size, cpu, track, smooth):
    net = net.eval()  # 锁定网络参数
    if not cpu:
        net = net.cuda()  # 启动GPU

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts  # 18个采样点
    previous_poses = []  # 预测集合

    orig_img = img.copy()
    heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)
    if track:
        track_poses(previous_poses, current_poses, smooth=smooth)
        previous_poses = current_poses

    if len(previous_poses) == 0:
        return []
    else:
        return previous_poses[0].keypoints
# 将cv图像显示在Tk组件上
def Showimage(imgCV_in, canva, layout="null"):
    global imgTK
    canvawidth = int(canva.winfo_reqwidth())
    canvaheight = int(canva.winfo_reqheight())
    sp = imgCV_in.shape
    cvheight = sp[0]  # height(rows) of image
    cvwidth = sp[1]  # width(colums) of image
    if (layout == "fill"):
        imgCV = cv2.resize(imgCV_in, (canvawidth, canvaheight), interpolation=cv2.INTER_AREA)
    elif (layout == "fit"):
        if (float(cvwidth / cvheight) > float(canvawidth / canvaheight)):
            imgCV = cv2.resize(imgCV_in, (canvawidth, int(canvawidth * cvheight / cvwidth)),
                               interpolation=cv2.INTER_AREA)
        else:
            imgCV = cv2.resize(imgCV_in, (int(canvaheight * cvwidth / cvheight), canvaheight),
                               interpolation=cv2.INTER_AREA)
    else:
        imgCV = imgCV_in
    imgCV2 = cv2.cvtColor(imgCV, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
    current_image = Image.fromarray(imgCV2)  # 将图像转换成Image对象
    imgTK = ImageTk.PhotoImage(image=current_image)  # 将image对象转换为imageTK对象
    canva.create_image(0, 0, anchor=NW, image=imgTK)
# 骨架计算距离函数
def clac_distance(a, b):
    dis_square = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    return math.sqrt(dis_square)
# 骨架计算角度函数
def clac_angel(a, b, c):
    return math.atan2((a[0] - b[0]), (a[1] - b[1])) - math.atan2((c[0] - b[0]), (c[1] - b[1]))
# 获取模型输入张量
def clac_keras(key_point):
    distance_all = []
    angel_all = []
    # 计算距离
    # 鼻子到脖子
    if key_point[0][0] != -1 and key_point[1][0] != -1:
        distance_all.append(clac_distance(key_point[0], key_point[1]))
    else:
        distance_all.append(avg[0])
    # 鼻子到右眼
    if key_point[0][0] != -1 and key_point[14][0] != -1:
        distance_all.append(clac_distance(key_point[0], key_point[14]))
    else:
        distance_all.append(avg[1])
    # 鼻子到左眼
    if key_point[0][0] != -1 and key_point[15][0] != -1:
        distance_all.append(clac_distance(key_point[0], key_point[15]))
    else:
        distance_all.append(avg[2])
    # 右眼到右耳
    if key_point[14][0] != -1 and key_point[16][0] != -1:
        distance_all.append(clac_distance(key_point[14], key_point[16]))
    else:
        distance_all.append(avg[3])
    # 左眼到左耳
    if key_point[15][0] != -1 and key_point[17][0] != -1:
        distance_all.append(clac_distance(key_point[15], key_point[17]))
    else:
        distance_all.append(avg[4])
    # 脖子到右肩
    if key_point[1][0] != -1 and key_point[2][0] != -1:
        distance_all.append(clac_distance(key_point[1], key_point[2]))
    else:
        distance_all.append(avg[5])
    # 右肩到右肘
    if key_point[2][0] != -1 and key_point[3][0] != -1:
        distance_all.append(clac_distance(key_point[2], key_point[3]))
    else:
        distance_all.append(avg[6])
    # 右肘到右腕
    if key_point[3][0] != -1 and key_point[4][0] != -1:
        distance_all.append(clac_distance(key_point[3], key_point[4]))
    else:
        distance_all.append(avg[7])
    # 脖子到左肩
    if key_point[1][0] != -1 and key_point[5][0] != -1:
        distance_all.append(clac_distance(key_point[1], key_point[5]))
    else:
        distance_all.append(avg[8])
    # 左肩到左肘
    if key_point[5][0] != -1 and key_point[6][0] != -1:
        distance_all.append(clac_distance(key_point[5], key_point[6]))
    else:
        distance_all.append(avg[9])
    # 左肘到左腕
    if key_point[6][0] != -1 and key_point[7][0] != -1:
        distance_all.append(clac_distance(key_point[6], key_point[7]))
    else:
        distance_all.append(avg[10])
    # 脖子到右臀
    if key_point[1][0] != -1 and key_point[8][0] != -1:
        distance_all.append(clac_distance(key_point[1], key_point[8]))
    else:
        distance_all.append(avg[11])
    # 右臀到右膝
    if key_point[8][0] != -1 and key_point[9][0] != -1:
        distance_all.append(clac_distance(key_point[8], key_point[9]))
    else:
        distance_all.append(avg[12])
    # 右膝到右踝
    if key_point[9][0] != -1 and key_point[10][0] != -1:
        distance_all.append(clac_distance(key_point[9], key_point[10]))
    else:
        distance_all.append(avg[13])
    # 脖子到左臀
    if key_point[1][0] != -1 and key_point[11][0] != -1:
        distance_all.append(clac_distance(key_point[1], key_point[11]))
    else:
        distance_all.append(avg[14])
    # 右臀到左膝
    if key_point[11][0] != -1 and key_point[12][0] != -1:
        distance_all.append(clac_distance(key_point[11], key_point[12]))
    else:
        distance_all.append(avg[15])
    # 右膝到左踝
    if key_point[12][0] != -1 and key_point[13][0] != -1:
        distance_all.append(clac_distance(key_point[12], key_point[13]))
    else:
        distance_all.append(avg[16])
    # 计算角度
    # 鼻子-右眼-右耳
    if key_point[0][0] != -1 and key_point[14][0] != -1 and key_point[16][0] != -1:
        angel_all.append(clac_angel(key_point[0], key_point[14], key_point[16]))
    else:
        angel_all.append(avg[17])
    # 鼻子-左眼-左耳
    if key_point[0][0] != -1 and key_point[15][0] != -1 and key_point[17][0] != -1:
        angel_all.append(clac_angel(key_point[0], key_point[15], key_point[17]))
    else:
        angel_all.append(avg[18])
    # 脖子-右肩-右肘
    if key_point[1][0] != -1 and key_point[2][0] != -1 and key_point[3][0] != -1:
        angel_all.append(clac_angel(key_point[1], key_point[2], key_point[3]))
    else:
        angel_all.append(avg[19])
    # 右肩-右肘-右腕
    if key_point[2][0] != -1 and key_point[3][0] != -1 and key_point[4][0] != -1:
        angel_all.append(clac_angel(key_point[2], key_point[3], key_point[4]))
    else:
        angel_all.append(avg[20])
    # 脖子-左肩-左肘
    if key_point[1][0] != -1 and key_point[5][0] != -1 and key_point[6][0] != -1:
        angel_all.append(clac_angel(key_point[1], key_point[5], key_point[6]))
    else:
        angel_all.append(avg[21])
    # 左肩-左肘-左腕
    if key_point[5][0] != -1 and key_point[6][0] != -1 and key_point[7][0] != -1:
        angel_all.append(clac_angel(key_point[5], key_point[6], key_point[7]))
    else:
        angel_all.append(avg[22])
    # 脖子-右臀-右膝
    if key_point[1][0] != -1 and key_point[8][0] != -1 and key_point[9][0] != -1:
        angel_all.append(clac_angel(key_point[1], key_point[8], key_point[9]))
    else:
        angel_all.append(avg[23])
    # 右臀-右膝-右踝
    if key_point[8][0] != -1 and key_point[9][0] != -1 and key_point[10][0] != -1:
        angel_all.append(clac_angel(key_point[8], key_point[9], key_point[10]))
    else:
        angel_all.append(avg[24])
    # 脖子-左臀-左膝
    if key_point[1][0] != -1 and key_point[11][0] != -1 and key_point[12][0] != -1:
        angel_all.append(clac_angel(key_point[1], key_point[11], key_point[12]))
    else:
        angel_all.append(avg[25])
    # 左臀-左膝-左踝
    if key_point[11][0] != -1 and key_point[12][0] != -1 and key_point[13][0] != -1:
        angel_all.append(clac_angel(key_point[11], key_point[12], key_point[13]))
    else:
        angel_all.append(avg[26])
    # 鼻子-脖子-右肩
    if key_point[0][0] != -1 and key_point[1][0] != -1 and key_point[2][0] != -1:
        angel_all.append(clac_angel(key_point[0], key_point[1], key_point[2]))
    else:
        angel_all.append(avg[27])
    # 鼻子-脖子-左肩
    if key_point[0][0] != -1 and key_point[1][0] != -1 and key_point[5][0] != -1:
        angel_all.append(clac_angel(key_point[0], key_point[1], key_point[5]))
    else:
        angel_all.append(avg[28])
    # 右眼-鼻子-左眼
    if key_point[14][0] != -1 and key_point[0][0] != -1 and key_point[15][0] != -1:
        angel_all.append(clac_angel(key_point[14], key_point[0], key_point[15]))
    else:
        angel_all.append(avg[29])
    # 右眼-鼻子-脖子
    if key_point[14][0] != -1 and key_point[0][0] != -1 and key_point[1][0] != -1:
        angel_all.append(clac_angel(key_point[14], key_point[0], key_point[1]))
    else:
        angel_all.append(avg[30])
    # 左眼-鼻子-脖子
    if key_point[15][0] != -1 and key_point[0][0] != -1 and key_point[1][0] != -1:
        angel_all.append(clac_angel(key_point[15], key_point[0], key_point[1]))
    else:
        angel_all.append(avg[31])
    # 鼻子-脖子-右臀
    if key_point[0][0] != -1 and key_point[1][0] != -1 and key_point[8][0] != -1:
        angel_all.append(clac_angel(key_point[0], key_point[1], key_point[8]))
    else:
        angel_all.append(avg[32])
    # 鼻子-脖子-左臀
    if key_point[0][0] != -1 and key_point[1][0] != -1 and key_point[11][0] != -1:
        angel_all.append(clac_angel(key_point[0], key_point[1], key_point[11]))
    else:
        angel_all.append(avg[33])
    data = distance_all + angel_all
    datas.loc[0] = data
    for i in [10 + 16, 8 + 16, 2 + 16, 16, 13, 4]:
        data.pop(i)
    return data
# 疲劳检测函数
def main_detect(cap):
    while switch == 1:
        start = cv2.getTickCount()
        result_show.grid_forget()
        canva_r.delete("all")
        global t, eye_close, yawn, yawn_flag
        ret, frame = cap.read()  # 读取摄像头 大小：(480x640)
        frame = frame[0:1080, 0:1920 - 480]
        frame = cv2.resize(frame, (int(frame.shape[1] / 2.25), int(frame.shape[0] / 2.25)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
        subjects = detect(gray, 0)
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)  # 获得68个特征点的坐标

            # 计算左右眼平均眼宽比
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftRatio = eye_aspect_ratio(leftEye)
            rightRatio = eye_aspect_ratio(rightEye)
            EyeRatio = (leftRatio + rightRatio) / 2.0

            # 计算嘴巴宽高比
            mouth = shape[mStart:mEnd]
            mouthRatio = mouth_aspect_ratio(mouth)

            # 画出凸包
            # leftEyeHull = cv2.convexHull(leftEye)
            # rightEyeHull = cv2.convexHull(rightEye)
            # mouthHull = cv2.convexHull(mouth)
            # cv2.drawContours(frame, [leftEyeHull], -1, (50, 50, 250), 2)
            # cv2.drawContours(frame, [rightEyeHull], -1, (50, 50, 250), 2)
            # cv2.drawContours(frame, [mouthHull], -1, (150, 50, 150), 2)

            # 判断是否打哈欠
            if mouthRatio > thresh_mouth:
                yawn = True
                yawn_flag = 0
            if yawn == True and yawn_flag < 40:
                canva_r.create_text(200, 200, text="检测到您打了一个哈欠，\n请注意不要疲劳驾驶！", font=("Lucida Console", 15), fill="red")
                if yawn == True and t == 0:
                    t = 1
                    pygame.mixer.music.stop()
                    pygame.mixer.music.load('sound\\yawn.mp3')
                    pygame.mixer.music.play()
                yawn_flag = yawn_flag + 1
            elif yawn == True and yawn_flag == 40:
                yawn = False
                yawn_flag = 0
                t = 0

            # 判断是否闭上眼睛
            if EyeRatio < thresh_eye:
                flag = flag + 1
                if flag >= frame_check:
                    eye_close = True
                    eye_flag = 0
            else:
                flag = 0
            if eye_close == True and eye_flag < 40:
                # WARNING
                canva_r.create_text(200, 200, text="警告！！！\n检测到您的眼睛已经闭合，\n请注意不要疲劳驾驶！", justify=LEFT,
                                    font=("Lucida Console", 15), fill="red")
                if eye_close == True and t == 0:
                    t = 1
                    pygame.mixer.music.stop()
                    pygame.mixer.music.load('sound\\eyes.mp3')
                    pygame.mixer.music.play()
                eye_flag = eye_flag + 1
            elif eye_close == True and eye_flag == 40:
                eye_close = False
                eye_flag = 0
                t = 0
        end = cv2.getTickCount()
        during1 = (end - start) / cv2.getTickFrequency()
        # 计算代码运行的时间消耗，其中最后一个参数是时钟周期

        FPS.set("FPS:" + str(round(1 / during1, 2)))
        Showimage(frame, canva_l, "fit")
        root.update()
# 驾驶状态分类
def main_class(vc):
    # 视频读取参数
    c = 0  # 开始帧
    timeF = 100  # 视频帧计数间隔频率
    result_show.grid(row=1, column=1)
    while switch == 0:
        # image = cv2.imread('img_59235.jpg')
        rval, image = vc.read()
        if c % timeF == 0:
            start = cv2.getTickCount()
            image = image[:, 240:1920 - 240, :]
            image = cv2.resize(image, (int(image.shape[1] / 2.25), int(image.shape[0] / 2.25)))

            key_point = run_demo(net, image, image.shape[0] / 3, False, 1, 1)
            data = clac_keras(key_point)

            y_pred = etc.predict([data])
            y_pred_proba = etc.predict_proba([data])

            canvas = copy.deepcopy(image)
            Showimage(canvas, canva_l, "fill")

            canva_r.delete("all")
            # 创建分类标签
            text_all = ("安全驾驶       ", "用右手发短信   ", "用右手打电话   ", "用左手发短信   ", "用左手打电话   ",
                        "调音乐播放器    ", "喝水         ", "后面拿东西    ", "弄头发或化妆   ", "与乘客交谈    ")
            for i in range(10):
                canva_r.create_text(70, 36 * i + 20, text=text_all[i], font=("Lucida Console", 10))
                canva_r.create_rectangle(150, 15 + 36 * i, 150 + 100 * y_pred_proba[0][i], 25 + 36 * i, fill='cyan')
                canva_r.create_text(300, 36 * i + 20, text=y_pred_proba[0][i], justify=LEFT)

            end = cv2.getTickCount()
            during1 = (end - start) / cv2.getTickFrequency()
            # 计算代码运行的时间消耗，其中最后一个参数是时钟周期
            FPS.set("FPS:" + str(round(1 / during1, 2)))
            result.set("识别结果为：" + text_all[y_pred[0]])
            root.update()
        # c = c + 1
# 按钮状态
def swi():
    global switch
    switch = not switch
# GUI初始化
def GUI_init():
    global result_show, canva_r, canva_l, FPS, root, result,switch
    # 创建GUI
    root = Tk()
    root.title("驾驶员检测")
    root.minsize(710, 410)
    # 创建视频幕布
    canva_l = Canvas(root, width=480, height=360, bg="white")
    canva_l.grid(row=0, column=0)
    # 创建概率直方图幕布
    canva_r = Canvas(root, width=350, height=360, bg="white")
    canva_r.grid(row=0, column=1)
    # 显示FPS
    FPS = tkinter.StringVar()
    FPS_show = tkinter.Label(root, textvariable=FPS, bg="white", font=("Lucida Console", 10))
    FPS_show.grid(row=1, column=0)
    # 显示识别结果
    result = tkinter.StringVar()
    result_show = tkinter.Label(root, textvariable=result, bg="white", font=("Lucida Console", 14))
    result_show.grid(row=1, column=1)
    # 创建切换按钮
    cut = tkinter.Button(root, text="切换视角", command=swi, font=("Lucida Console", 14))
    cut.place(x=350, y=366)