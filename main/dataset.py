# Copyright (c) OpenMMLab. All rights reserved.

# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import logging
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmfewshot.detection.datasets.coco import FewShotCocoDataset
from terminaltables import AsciiTable

from .split import SPLIT
    
work_root = osp.dirname(__file__)
data_root = f'{work_root}/../dataset/'

@DATASETS.register_module()
class FewShotDefectDataset(FewShotCocoDataset):
    def __init__(self,
                classes: Optional[Union[str, Sequence[str]]] = None,
                num_novel_shots: Optional[int] = None,
                num_base_shots: Optional[int] = None,
                ann_shot_filter: Optional[Dict] = None,    
                dataset_name: Optional[str] = None, 
                test_mode: bool = False,
                defect_name: Optional[str] = None, 
                **kwargs) -> None:
        
        self.SPLIT = SPLIT[defect_name]
        self.num_base_shots = num_base_shots
        self.num_novel_shots = num_novel_shots
        if dataset_name is None:
            self.dataset_name = 'Test dataset' \
                if test_mode else 'Train dataset'
        else:
            self.dataset_name = dataset_name
        self.CLASSES = self.get_classes(classes)  
        if ann_shot_filter is None:
            if num_novel_shots is not None or num_base_shots is not None:
                ann_shot_filter = self._create_ann_shot_filter()
        else:
            assert num_novel_shots is None and num_base_shots is None, \
                f'{self.dataset_name}: can not config ann_shot_filter and ' \
                f'num_novel_shots/num_base_shots at the same time.'
        super().__init__(
            classes=classes,
            ann_shot_filter=ann_shot_filter,
            **kwargs)

    def get_classes(self, classes: Union[str, Sequence[str]]) -> List[str]:
        """Get class names.

        It supports to load pre-defined classes splits.
        The pre-defined classes splits are:

        Args:
            classes (str | Sequence[str]): Classes for model training and
                provide fixed label for each class. When classes is string,
                it will load pre-defined classes in `FewShotVOCDataset`.
                For example: 'NOVEL_CLASSES'.

        Returns:
            list[str]: List of class names.
        """
        # configure few shot classes setting
        if isinstance(classes, str):
            assert classes in self.SPLIT.keys(
            ), f'{self.dataset_name}: not a pre-defined classes or ' \
               f'split in VOC_SPLIT'
            class_names = self.SPLIT[classes]
            if 'BASE_CLASSES' in classes:
                assert self.num_novel_shots is None, \
                    f'{self.dataset_name}: BASE_CLASSES do not have ' \
                    f'novel instances.'
            elif 'NOVEL_CLASSES' in classes:
                assert self.num_base_shots is None, \
                    f'{self.dataset_name}: NOVEL_CLASSES do not have ' \
                    f'base instances.'
            self.split_id = int(classes[-1])
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def _create_ann_shot_filter(self) -> Dict[str, int]:
        """Generate `ann_shot_filter` for novel and base classes.

        Returns:
            dict[str, int]: The number of shots to keep for each class.
        """
        ann_shot_filter = {}
        if self.num_novel_shots is not None:
            for class_name in self.SPLIT[
                    f'NOVEL_CLASSES_SPLIT{self.split_id}']:
                ann_shot_filter[class_name] = self.num_novel_shots
        if self.num_base_shots is not None:
            for class_name in self.SPLIT[f'BASE_CLASSES_SPLIT{self.split_id}']:
                ann_shot_filter[class_name] = self.num_base_shots
        return ann_shot_filter

    
@DATASETS.register_module()
class FewShotDefectCopyDataset(FewShotDefectDataset):
    """Copy other VOC few shot datasets' `data_infos` directly.

    This dataset is mainly used for model initialization in some meta-learning
    detectors. In their cases, the support data are randomly sampled
    during training phase and they also need to be used in model
    initialization before evaluation. To copy the random sampling results,
    this dataset supports to load `data_infos` of other datasets via `ann_cfg`

    Args:
        ann_cfg (list[dict] | dict): contain `data_infos` from other
            dataset. Example: [dict(data_infos=FewShotVOCDataset.data_infos)]
    """

    def __init__(self, ann_cfg: Union[List[Dict], Dict], **kwargs) -> None:
        super().__init__(ann_cfg=ann_cfg, **kwargs)

    def ann_cfg_parser(self, ann_cfg: Union[List[Dict], Dict]) -> List[Dict]:
        """Parse annotation config from a copy of other dataset's `data_infos`.

        Args:
            ann_cfg (list[dict] | dict): contain `data_infos` from other
                dataset. Example:
                [dict(data_infos=FewShotVOCDataset.data_infos)]

        Returns:
            list[dict]: Annotation information.
        """
        data_infos = []
        if isinstance(ann_cfg, dict):
            assert ann_cfg.get('data_infos', None) is not None, \
                f'{self.dataset_name}: ann_cfg of ' \
                f'FewShotVOCCopyDataset require data_infos.'
            # directly copy data_info
            data_infos = ann_cfg['data_infos']
        elif isinstance(ann_cfg, list):
            for ann_cfg_ in ann_cfg:
                assert ann_cfg_.get('data_infos', None) is not None, \
                    f'{self.dataset_name}: ann_cfg of ' \
                    f'FewShotVOCCopyDataset require data_infos.'
                # directly copy data_info
                data_infos += ann_cfg_['data_infos']
        return data_infos

@DATASETS.register_module()
class FewShotDefectDefaultDataset(FewShotDefectDataset):

    def __init__(self, ann_cfg: List[Dict], **kwargs) -> None:

        self.DEFAULT_ANN_CONFIG = dict(
            NEU_DET={},
            # GC10_DET={},
            DeepPCB={})
        
        for dataset in self.DEFAULT_ANN_CONFIG.keys():
            self.DEFAULT_ANN_CONFIG[dataset] = {
                f'SPLIT{split}_SEED{seed}_{shot}SHOT': [
                    dict(
                        type='ann_file',
                        ann_file = f'{data_root}/{dataset}/annotations/fewshot-split/'
                        f'/{seed}/{seed}seed_{shot}shot_{class_name}_trainval.json')
                    for class_name in SPLIT[dataset][f'ALL_CLASSES_SPLIT{split}']
                ]
                for split in [1,2,3]
                for shot in [5,10,30] for seed in range(1,10)}

        super().__init__(ann_cfg=ann_cfg,defect_name=ann_cfg[0]["dataset_name"], **kwargs)
        
    def ann_cfg_parser(self, ann_cfg: List[Dict]) -> List[Dict]:
        
        new_ann_cfg = []
        for ann_cfg_ in ann_cfg:
            assert isinstance(ann_cfg_, dict), \
                f'{self.dataset_name}: ann_cfg should be list of dict.'
            dataset_name = ann_cfg_['dataset_name']
            setting = ann_cfg_['setting']
            default_ann_cfg = self.DEFAULT_ANN_CONFIG[dataset_name][setting]
            ann_root = ann_cfg_.get('ann_root', None)
            if ann_root is not None:
                for i in range(len(default_ann_cfg)):
                    default_ann_cfg[i]['ann_file'] = osp.join(
                        ann_root, default_ann_cfg[i]['ann_file'])
            new_ann_cfg += default_ann_cfg
        # print(new_ann_cfg)
        return super(FewShotDefectDataset, self).ann_cfg_parser(new_ann_cfg)