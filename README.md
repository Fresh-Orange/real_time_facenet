# real_time_facenet
Detect and identify the face in camera using [facenet](https://github.com/davidsandberg/facenet), which is a tensorFlow implementation of the face recognizer described in the paper
["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832). All I did was extract what I thought was the most useful file from the original project and add the camera data extraction code to the original file. For more detail about facenet, you'd better read [original project](https://github.com/davidsandberg/facenet)

将[facenet](https://github.com/davidsandberg/facenet)模型用于做摄像头的人脸检测和识别, 它是一个人脸识别模型的tensorflow实现版本，具体的原理请看论文["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832). 我做的只是从原来的facenet代码中提取出我认为最有用的几个代码文件并且在代码中加入获取摄像头数据作为模型输入的代码，原项目有更多文件，你可以去阅读[原项目](https://github.com/davidsandberg/facenet)

## Pre-trained models(预训练模型)
| Model download      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [Google_drive](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) or [BaiduYun](https://pan.baidu.com/s/1k_k6I0d85T7yVLRFovkj3w) | 0.9905        | CASIA-WebFace    | Inception ResNet v1|
| [Google_drive](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965        | VGGFace2      | Inception ResNet v1 |

| 模型下载      | LFW 准确率 | 训练数据 | 网络架构 |
|-----------------|--------------|------------------|-------------|
| [谷歌云](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) or [百度云](https://pan.baidu.com/s/1k_k6I0d85T7yVLRFovkj3w) | 0.9905        | CASIA-WebFace    | Inception ResNet v1|
| [谷歌云](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965        | VGGFace2      | Inception ResNet v1 |

## How to use the project(如何使用这个项目)
1. Download any one of the pretrained facenet model. （下载上述任意一个预先训练模型。）
2. Place your face dataset in `dataset_without_align` directory. （将你自己的人脸数据集放在`dataset_without_align`文件夹.）
3. Modify the model_dir in `train_predict.py`, then run this file to train model and predict. （修改`train_predict.py`里面的模型路径(model_dir), 然后运行这个文件即可）

