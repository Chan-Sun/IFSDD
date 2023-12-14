_base_ = [
    '../_base_/datasets/ft_base.py',
    '../_base_/models/faster_rcnn_r50_caffe_fpn.py',
    '../_base_/base_settings.py',
]

data = dict(
    train=dict(classes='BASE_CLASSES_SPLIT1'),
    val=dict(classes='BASE_CLASSES_SPLIT1'),
    test=dict(classes='BASE_CLASSES_SPLIT1'))


# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=3)))
