# 基于OpenPose的驾驶员检测系统

>>效果展示  
<img src="https://github.com/Dou-noki/Driver-detection-based-on-OpenPose-and-RandomForest/blob/main/image/1.jpg?raw=true" width="200" height="200"/>
![Image text](https://github.com/Dou-noki/Driver-detection-based-on-OpenPose-and-RandomForest/blob/main/image/1.jpg?raw=true)
![Image text](https://github.com/Dou-noki/Driver-detection-based-on-OpenPose-and-RandomForest/blob/main/image/3.jpg?raw=true)
![Image text](https://github.com/Dou-noki/Driver-detection-based-on-OpenPose-and-RandomForest/blob/main/image/11.jpg?raw=true)
![Image text](https://github.com/Dou-noki/Driver-detection-based-on-OpenPose-and-RandomForest/blob/main/image/12.jpg?raw=true)

main.py 为程序执行入口

get_train.py 为OpenPose提取姿态特征文件，数据集来源为Kaggle网站上State Farm Distracted Driver Detection
地址：https://www.kaggle.com/c/state-farm-distracted-driver-detection

func.py 为该程序的函数支持库，具有中文详细注释

val.py 、modules与datasets为Light-OpenPose支持文件

model中保存了已训练好的OpenPose、dlib与随机森林的模型文件

video文件夹中放置侧置摄像头视频与前置摄像头视频

需要安装的库不少，建议边看报错提示，边安装。
