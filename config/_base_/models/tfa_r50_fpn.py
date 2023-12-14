_base_ = ['./faster_rcnn_r50_caffe_fpn.py']
                    
model = dict(
    type='TFA',
    frozen_parameters=[
        'backbone','neck', 'rpn_head', 'roi_head.bbox_head.shared_fcs'
    ],
    roi_head=dict(
        bbox_head=dict(
            type='CosineSimBBoxHead',
            num_shared_fcs=2,
            num_classes=6,
            scale=20)))
