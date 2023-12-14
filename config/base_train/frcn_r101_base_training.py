_base_ = ["./frcn_r50_base_training.py"]



# model settings
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101))
