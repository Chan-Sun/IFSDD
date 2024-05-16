_base_ = ['./tfa_r50_fpn.py']

model = dict(
    type='TFA',
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101, frozen_stages=4))
