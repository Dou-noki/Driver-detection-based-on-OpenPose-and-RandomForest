## 基于OpenPose与随机森林算法的驾驶员状态检测系统

>效果展示  
<img src="https://github.com/Dou-noki/Driver-detection-based-on-OpenPose-and-RandomForest/blob/main/image/1.jpg?raw=true" width="420" height="224"/>
<img src="https://github.com/Dou-noki/Driver-detection-based-on-OpenPose-and-RandomForest/blob/main/image/12.jpg?raw=true" width="420" height="224"/>

main.py 为程序执行入口

get_train.py 为OpenPose提取姿态特征文件，数据集来源为Kaggle网站上State Farm Distracted Driver Detection
地址：https://www.kaggle.com/c/state-farm-distracted-driver-detection
输出为openpose_train_data.csv文件

func.py 为该程序的函数支持库，具有中文详细注释

val.py 、modules与datasets为Light-OpenPose支持文件

model中需放置已训练好的OpenPose、dlib与随机森林的3个模型文件
模型下载：

链接：https://pan.baidu.com/s/1qMid2zZWTuaPjE2nIkLWkw?pwd=0vzs 
提取码：0vzs 

video文件夹中放置侧置摄像头视频与前置摄像头视频

sound文件夹中为疲劳检测语音提示音

image文件中为程序运行效果

需要安装的库不少，如果有缺少的，建议边看报错的提示边安装。
