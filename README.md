# Weapon detection project.
[![Clickable](rep_images/preview_v2.png)](https://youtu.be/bor4XYdBg1k)
Click on the picture to view a demonstration of the model's work on some videos from YouTube.\
The first version of the model was trained on 6427 images over 100 epochs.\
**Stack:** YOLOv8, OpenCV. 
### Score
**mAP50:** 0.942\
**Precision:** 0.918\
**Recall:** 0.899
### Dataset
The [original](https://arxiv.org/abs/2105.01058) dataset was created by Delong Qi, Weijun Tan, Zhifu Liu, Qi Yao, Jingfeng Liu.\
Some background pictures, augmentation and screenshots from real videos on YouTube were added to it.\
The data includes 6427 train (81.14%), 997 val (12.59%) and 497 test (6.27%) images.\
In the future, the model will be trained on the whole dataset (51K images).\
**Image examples:**
<p align="center">
<img src="rep_images/train_batch36181.jpg"></p>

### Project files
```
.
├── gun_detection_yolo.pt # Weights for YOLOv8 model
├── video_live.py         # Using a model for a recorded video with live result
├── video_processing.py   # Using a model for a recorded video and save result
└── webcam_live.py        # Using a model for a webcam with live result
```