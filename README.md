# real_time_facenet
Detect and identify the face in camera using [facenet](https://github.com/davidsandberg/facenet), which is a tensorFlow implementation of the face recognizer described in the paper
["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832). All I did was extract what I thought was the most useful file from the original project and add the camera data extraction code to the original file. For more detail about facenet, you'd better read [original project](https://github.com/davidsandberg/facenet)


## Pre-trained models
| Model download      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [Google_drive](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) [BaiduYun](https://pan.baidu.com/s/1k_k6I0d85T7yVLRFovkj3w) | 0.9905        | CASIA-WebFace    | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| [Google_drive](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965        | VGGFace2      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |

## How to use the project
First, download any one of the pretrained facenet model. Second, place your face dataset in `dataset_without_align` directory. And then ...
| OS    | method |
|-----------------|--------------|
| Windows | Modify the model directory path in `train_predict.bat`, then double click it to train model and predict|
| Linux | Open `train_predic.bat` and you'll know how to run those commands since you are suing linux |


