import argparse

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale) # 标准化图片

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

def run_demo(net, img, height_size, cpu, track, smooth):
    net = net.eval() # 锁定网络参数
    if not cpu:
        net = net.cuda() # 启动GPU

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts # 18个采样点
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

    if len(previous_poses)==0:
        return []
    else:
        return previous_poses[0].keypoints



BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

BODY_PARTS = { "鼻子": 0, "脖子": 1,
               "右肩": 2, "右肘": 3, "右腕": 4,
               "左肩": 5, "左肘": 6, "左腕": 7,
               "右臀": 8, "右膝": 9, "右踝": 10,
               "左臀": 11,"左膝": 12,"左踝": 13,
               "右眼": 14,"左眼": 15,
               "右耳": 16,"左耳": 17 }
import math
import pandas as pd
import os

# 计算距离函数
def distance(a,b):
    dis_square = (a[0]-b[0])**2 + (a[1]-b[1])**2
    return math.sqrt(dis_square)

# 计算角度函数
def angel(a,b,c):
    return math.atan2((a[0]-b[0]),(a[1]-b[1]))-math.atan2((c[0]-b[0]),(c[1]-b[1]))


if __name__ == '__main__':
    net = PoseEstimationWithMobileNet() # 加载网络结构
    checkpoint = torch.load('checkpoint_iter_370000.pth', map_location='cpu') # 加载模型参数
    load_state(net, checkpoint) # 拼接结构与参数

    datas = pd.DataFrame(
        columns=['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15', 'd16',
                 'd17',
                 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16',
                 'a17', 'label'])

    for label_name in sorted(os.listdir("../data/imgs/train")):
        print(label_name)
        for image_name in sorted(os.listdir("../data/imgs/train/" + label_name)):
            print(image_name)
            image = cv2.imread(os.path.join("../data/imgs/train/" + label_name, image_name))

            # print (image.shape) # (480, 640, 3)
            key_point = run_demo(net, image, image.shape[0]/3, False, 1, 1)
            distance_all=[]
            angel_all=[]
            if len(key_point)==0:
                continue
            else:
                # 计算距离
                # 鼻子到脖子
                if key_point[0][0]!=-1 and key_point[1][0]!=-1:
                    distance_all.append(distance(key_point[0], key_point[1]))
                else:
                    distance_all.append(np.nan)
                # 鼻子到右眼
                if key_point[0][0]!=-1 and key_point[14][0]!=-1:
                    distance_all.append(distance(key_point[0], key_point[14]))
                else:
                    distance_all.append(np.nan)
                # 鼻子到左眼
                if key_point[0][0]!=-1 and key_point[15][0]!=-1:
                    distance_all.append(distance(key_point[0], key_point[15]))
                else:
                    distance_all.append(np.nan)
                # 右眼到右耳
                if key_point[14][0]!=-1 and key_point[16][0]!=-1:
                    distance_all.append(distance(key_point[14], key_point[16]))
                else:
                    distance_all.append(np.nan)
                # 左眼到左耳
                if key_point[15][0]!=-1 and key_point[17][0]!=-1:
                    distance_all.append(distance(key_point[15], key_point[17]))
                else:
                    distance_all.append(np.nan)
                # 脖子到右肩
                if key_point[1][0]!=-1 and key_point[2][0]!=-1:
                    distance_all.append(distance(key_point[1], key_point[2]))
                else:
                    distance_all.append(np.nan)
                # 右肩到右肘
                if key_point[2][0]!=-1 and key_point[3][0]!=-1:
                    distance_all.append(distance(key_point[2], key_point[3]))
                else:
                    distance_all.append(np.nan)
                # 右肘到右腕
                if key_point[3][0]!=-1 and key_point[4][0]!=-1:
                    distance_all.append(distance(key_point[3], key_point[4]))
                else:
                    distance_all.append(np.nan)
                # 脖子到左肩
                if key_point[1][0]!=-1 and key_point[5][0]!=-1:
                    distance_all.append(distance(key_point[1], key_point[5]))
                else:
                    distance_all.append(np.nan)
                # 左肩到左肘
                if key_point[5][0]!=-1 and key_point[6][0]!=-1:
                    distance_all.append(distance(key_point[5], key_point[6]))
                else:
                    distance_all.append(np.nan)
                # 左肘到左腕
                if key_point[6][0]!=-1 and key_point[7][0]!=-1:
                    distance_all.append(distance(key_point[6], key_point[7]))
                else:
                    distance_all.append(np.nan)
                # 脖子到右臀
                if key_point[1][0]!=-1 and key_point[8][0]!=-1:
                    distance_all.append(distance(key_point[1], key_point[8]))
                else:
                    distance_all.append(np.nan)
                # 右臀到右膝
                if key_point[8][0]!=-1 and key_point[9][0]!=-1:
                    distance_all.append(distance(key_point[8], key_point[9]))
                else:
                    distance_all.append(np.nan)
                # 右膝到右踝
                if key_point[9][0]!=-1 and key_point[10][0]!=-1:
                    distance_all.append(distance(key_point[9], key_point[10]))
                else:
                    distance_all.append(np.nan)
                # 脖子到左臀
                if key_point[1][0]!=-1 and key_point[11][0]!=-1:
                    distance_all.append(distance(key_point[1], key_point[11]))
                else:
                    distance_all.append(np.nan)
                # 右臀到左膝
                if key_point[11][0]!=-1 and key_point[12][0]!=-1:
                    distance_all.append(distance(key_point[11], key_point[12]))
                else:
                    distance_all.append(np.nan)
                # 右膝到左踝
                if key_point[12][0]!=-1 and key_point[13][0]!=-1:
                    distance_all.append(distance(key_point[12], key_point[13]))
                else:
                    distance_all.append(np.nan)
                #计算角度
                # 鼻子-右眼-右耳
                if key_point[0][0]!=-1 and key_point[14][0]!=-1 and key_point[16][0]!=-1:
                    angel_all.append(angel(key_point[0],key_point[14],key_point[16]))
                else:
                    angel_all.append(np.nan)
                # 鼻子-左眼-左耳
                if key_point[0][0] != -1 and key_point[15][0] != -1 and key_point[17][0] != -1:
                    angel_all.append(angel(key_point[0], key_point[15], key_point[17]))
                else:
                    angel_all.append(np.nan)
                # 脖子-右肩-右肘
                if key_point[1][0] != -1 and key_point[2][0] != -1 and key_point[3][0] != -1:
                    angel_all.append(angel(key_point[1], key_point[2], key_point[3]))
                else:
                    angel_all.append(np.nan)
                # 右肩-右肘-右腕
                if key_point[2][0] != -1 and key_point[3][0] != -1 and key_point[4][0] != -1:
                    angel_all.append(angel(key_point[2], key_point[3], key_point[4]))
                else:
                    angel_all.append(np.nan)
                # 脖子-左肩-左肘
                if key_point[1][0] != -1 and key_point[5][0] != -1 and key_point[6][0] != -1:
                    angel_all.append(angel(key_point[1], key_point[5], key_point[6]))
                else:
                    angel_all.append(np.nan)
                # 左肩-左肘-左腕
                if key_point[5][0] != -1 and key_point[6][0] != -1 and key_point[7][0] != -1:
                    angel_all.append(angel(key_point[5], key_point[6], key_point[7]))
                else:
                    angel_all.append(np.nan)
                # 脖子-右臀-右膝
                if key_point[1][0] != -1 and key_point[8][0] != -1 and key_point[9][0] != -1:
                    angel_all.append(angel(key_point[1], key_point[8], key_point[9]))
                else:
                    angel_all.append(np.nan)
                # 右臀-右膝-右踝
                if key_point[8][0] != -1 and key_point[9][0] != -1 and key_point[10][0] != -1:
                    angel_all.append(angel(key_point[8], key_point[9], key_point[10]))
                else:
                    angel_all.append(np.nan)
                # 脖子-左臀-左膝
                if key_point[1][0] != -1 and key_point[11][0] != -1 and key_point[12][0] != -1:
                    angel_all.append(angel(key_point[1], key_point[11], key_point[12]))
                else:
                    angel_all.append(np.nan)
                # 左臀-左膝-左踝
                if key_point[11][0] != -1 and key_point[12][0] != -1 and key_point[13][0] != -1:
                    angel_all.append(angel(key_point[11], key_point[12], key_point[13]))
                else:
                    angel_all.append(np.nan)
                # 鼻子-脖子-右肩
                if key_point[0][0] != -1 and key_point[1][0] != -1 and key_point[2][0] != -1:
                    angel_all.append(angel(key_point[0], key_point[1], key_point[2]))
                else:
                    angel_all.append(np.nan)
                # 鼻子-脖子-左肩
                if key_point[0][0] != -1 and key_point[1][0] != -1 and key_point[5][0] != -1:
                    angel_all.append(angel(key_point[0], key_point[1], key_point[5]))
                else:
                    angel_all.append(np.nan)
                # 右眼-鼻子-左眼
                if key_point[14][0] != -1 and key_point[0][0] != -1 and key_point[15][0] != -1:
                    angel_all.append(angel(key_point[14], key_point[0], key_point[15]))
                else:
                    angel_all.append(np.nan)
                # 右眼-鼻子-脖子
                if key_point[14][0] != -1 and key_point[0][0] != -1 and key_point[1][0] != -1:
                    angel_all.append(angel(key_point[14], key_point[0], key_point[1]))
                else:
                    angel_all.append(np.nan)
                # 左眼-鼻子-脖子
                if key_point[15][0] != -1 and key_point[0][0] != -1 and key_point[1][0] != -1:
                    angel_all.append(angel(key_point[15], key_point[0], key_point[1]))
                else:
                    angel_all.append(np.nan)
                # 鼻子-脖子-右臀
                if key_point[0][0] != -1 and key_point[1][0] != -1 and key_point[8][0] != -1:
                    angel_all.append(angel(key_point[0], key_point[1], key_point[8]))
                else:
                    angel_all.append(np.nan)
                # 鼻子-脖子-左臀
                if key_point[0][0] != -1 and key_point[1][0] != -1 and key_point[11][0] != -1:
                    angel_all.append(angel(key_point[0], key_point[1], key_point[11]))
                else:
                    angel_all.append(np.nan)

            data = distance_all + angel_all + [label_name[1]]
            datas.loc[image_name] = data
    datas.to_csv("openpose_train_data.csv",sep=',')
