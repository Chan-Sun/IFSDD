import mmcv
import os
import torch
from torch.distributed import launch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel,MMDistributedDataParallel
from mmcv.runner import (get_dist_info, load_checkpoint,
                         wrap_fp16_model)
from mmdet.datasets import replace_ImageToTensor
from mmrazor.models import build_algorithm
from mmdet.utils import setup_multi_processes
import pickle
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path+"/../")
sys.path.append(current_path+"/../")
from mmfewshot.detection.datasets import build_dataloader, build_dataset,get_copy_dataset_type
from utils import ResultVisualizer,bbox_map_eval
from mmdet.datasets import get_loading_pipeline
from main import *

def model_output(cfg,ckpt_path,gpu_id=[0]):

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.algorithm.architecture.model.pretrained = None
    if cfg.algorithm.architecture.model.get('neck'):
        if isinstance(cfg.algorithm.architecture.model.neck, list):
            for neck_cfg in cfg.algorithm.architecture.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.algorithm.architecture.model.neck.get('rfp_backbone'):
            if cfg.algorithm.architecture.model.neck.rfp_backbone.get(
                    'pretrained'):
                cfg.algorithm.architecture.model.neck.rfp_backbone.pretrained = None  # noqa E501
    # pop frozen_parameters
    cfg.algorithm.architecture.model.pop('frozen_parameters', None)


    # build the algorithm and load checkpoint
    cfg.algorithm.architecture.model.train_cfg = None
    algorithm = build_algorithm(cfg.algorithm)
    
    model = algorithm.architecture.model 


    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        algorithm, ckpt_path, map_location='cpu')
    model = fuse_conv_bn(model)

    # in case the test dataset is concatenated
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(
            cfg.data.test.pipeline)
    cfg.gpu_ids = gpu_id
    rank, _ = get_dist_info()
    
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    model.CLASSES = dataset.CLASSES
    
    algorithm = MMDataParallel(algorithm, device_ids=cfg.gpu_ids)


    algorithm.eval()
    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = algorithm(return_loss=False, rescale=True, **data)
        batch_size = len(result)
        results.extend(result)
        for _ in range(batch_size):
            prog_bar.update()


    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule', 'dynamic_intervals'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric="bbox"))
    metric = dataset.evaluate(results, **eval_kwargs)
    print(metric)

    return results

def bbox_visualize(cfg,outputs,show_dir):
            
    cfg.data.test.test_mode = True
    os.makedirs(show_dir,exist_ok=True)
    cfg.data.test.pop('samples_per_gpu', 0)
    cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.pipeline)
    datasets = build_dataset(cfg.data.test)

    result_visualizer = ResultVisualizer(score_thr=0.3)
    

    prog_bar = mmcv.ProgressBar(len(outputs))
    _mAPs = {}
    for i, (result, ) in enumerate(zip(outputs)):
        # self.dataset[i] should not call directly
        # because there is a risk of mismatch
        data_info = datasets.prepare_train_img(i)
        mAP = bbox_map_eval(result, data_info['ann_info'])
        _mAPs[i] = mAP
        # _mAPs[i] = 1
        prog_bar.update()

    # descending select topk image
    _mAPs = list(sorted(_mAPs.items(), key=lambda kv: kv[1]))
    result_visualizer._save_image_gts_results(datasets, outputs, _mAPs, show_dir)

def save2json(cfg,results,save_path=None):
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset[idx]["img_metas"][0].data["ori_filename"]
        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_name'] = img_id
                data['bbox'] = [float(b) for b in bboxes[i]][:-1]
                data['score'] = round(float(bboxes[i][4]),4)
                data['category_id'] = label
                json_results.append(data)

    import json

    save_dict  ={"results":json_results}
    with open(save_path, 'w') as fp:
        json.dump(save_dict, fp,indent=4, separators=(',', ': '))
    
    return save_dict