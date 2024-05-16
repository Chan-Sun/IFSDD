_base_ = [
    "../_base_/datasets/ft_few_shot.py",
    '../_base_/fine_tune_settings.py',
    "../_base_/models/tfa_st.py"
]

student_class = 6
teacher_class = 6
shots=5
split=1

student = dict(
        init_cfg=dict(
            type='Pretrained',
            # checkpoint="/path/to/pretrain/model"
            checkpoint ="/home/chensun/Packages/IFSDD/testckpt_fine-tuning.pth"
        ),
        backbone=dict(frozen_stages=4),
        roi_head=dict(
            bbox_head=dict(
            type='CosineSimBBoxHead',
            num_classes=student_class,
            num_shared_fcs=2,
            scale=20
            )))

teacher = dict(        
        init_cfg=dict(
            type='Pretrained',
            # checkpoint="/path/to/pretrain/model"
            checkpoint ="/home/chensun/Packages/IFSDD/testckpt_fine-tuning.pth"

        ),
        roi_head=dict(
            bbox_head=dict(
                num_classes=teacher_class,
                )))

data = dict(
    train=dict(
        type='FewShotDefectDefaultDataset',
        ann_cfg=[dict(method='TFA', setting=f'SPLIT{split}_SEED1_{shots}SHOT')],
        num_novel_shots=shots,
        num_base_shots=shots,
        classes=f'ALL_CLASSES_SPLIT{split}',
        instance_wise=True),
    val=dict(classes=f'ALL_CLASSES_SPLIT{split}'),
    test=dict(classes=f'ALL_CLASSES_SPLIT{split}'))

weight=1
tau=5

distill_cfg = [
            dict(
                student_module='roi_head.bbox_head.fc_cls',
                teacher_module='roi_head.bbox_head.fc_cls',
                losses=[
                    dict(
                        type='LogitKnowAlignLoss',
                        name='loss_logits',
                        tau=tau,
                        loss_weight=0.01,
                        base_class=3)]),
            dict(
            student_module = 'neck.fpn_convs.3.conv',
            teacher_module = 'neck.fpn_convs.3.conv',
            losses=[
                dict(
                    type='FeatureKnowledgeAlignLoss',
                    name='loss_fpn_3',
                    tau = tau,
                    loss_weight =weight)]),
            dict(
            student_module = 'neck.fpn_convs.2.conv',
            teacher_module = 'neck.fpn_convs.2.conv',
            losses=[
                dict(
                    type='FeatureKnowledgeAlignLoss',
                    name='loss_fpn_2',
                    tau = tau,
                    loss_weight =weight)]),
            dict(
            student_module = 'neck.fpn_convs.1.conv',
            teacher_module = 'neck.fpn_convs.1.conv',
            losses=[
                dict(
                    type='FeatureKnowledgeAlignLoss',
                    name='loss_fpn_1',
                    tau = tau,
                    loss_weight =weight)]),
            dict(
            student_module = 'neck.fpn_convs.0.conv',
            teacher_module = 'neck.fpn_convs.0.conv',
            losses=[
                dict(
                    type='FeatureKnowledgeAlignLoss',
                    name='loss_fpn_0',
                    tau = tau,
                    loss_weight=weight)])
            ]  

# algorithm setting
algorithm = dict(
    architecture=dict(model=student),
    distiller=dict(
            teacher=teacher,
            components=distill_cfg))

find_unused_parameters = True
