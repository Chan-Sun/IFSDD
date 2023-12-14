import copy
import random
import warnings
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner)
from mmcv.utils import ConfigDict, build_from_cfg
from mmdet.core import DistEvalHook, EvalHook
from mmfewshot.detection.core import (QuerySupportDistEvalHook,
                                      QuerySupportEvalHook)
from mmfewshot.detection.datasets import build_dataloader, build_dataset
from mmfewshot.utils import get_root_logger
from mmrazor.core.distributed_wrapper import DistributedDataParallelWrapper
from mmrazor.core.hooks import DistSamplerSeedHook
from mmrazor.core.optimizer import build_optimizers


def get_copy_dataset_type(dataset_type: str) -> str:
    """Return corresponding copy dataset type."""
    if dataset_type in ['FewShotVOCDataset', 'FewShotVOCDefaultDataset']:
        copy_dataset_type = 'FewShotVOCCopyDataset'
    elif dataset_type in ['FewShotCocoDataset', 'FewShotCocoDefaultDataset']:
        copy_dataset_type = 'FewShotCocoCopyDataset'
    elif dataset_type in ['FewShotDefectDataset', 'FewShotDefectDefaultDataset']:
        copy_dataset_type = 'FewShotDefectCopyDataset'
    else:
        raise TypeError(f'{dataset_type} '
                        f'not support copy data_infos operation.')

    return copy_dataset_type

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set ``torch.backends.cudnn.deterministic``
            to True and ``torch.backends.cudnn.benchmark`` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_ifs_detector(model: nn.Module,
                   dataset: Iterable,
                   cfg: ConfigDict,
                   distributed: bool = False,
                   validate: bool = False,
                   timestamp: Optional[str] = None,
                   device='cuda',
                   meta: Optional[Dict] = None) -> None:
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loader = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            data_cfg=copy.deepcopy(cfg.data),
            use_infinite_sampler=cfg.use_infinite_sampler) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        if cfg.get('use_ddp_wrapper', False):
            # Difference from mmdetection.
            # In some algorithms, the ``optimizer.step()`` is executed in
            # ``train_step``. To rebuilt reducer buckets rightly, there need to
            # use DistributedDataParallelWrapper.
            model = DistributedDataParallelWrapper(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            # Sets the ``find_unused_parameters`` parameter in
            # torch.nn.parallel.DistributedDataParallel
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        if device == 'cuda':
            model = MMDataParallel(
                model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        elif device == 'cpu':
            model = model.cpu()
        else:
            raise ValueError(F'unsupported device name {device}.')

    if "algorithmn" in cfg.keys(): 
        optimizer = build_optimizers(model, cfg.optimizer)
    else:
        optimizer = build_optimizer(model, cfg.optimizer)

    # Infinite sampler will return a infinite stream of index. It can NOT
    # be used in `EpochBasedRunner`, because the `EpochBasedRunner` will
    # enumerate the dataloader forever. Thus, `InfiniteEpochBasedRunner`
    # is designed to handle dataloader with infinite sampler.
    if cfg.use_infinite_sampler and cfg.runner['type'] == 'EpochBasedRunner':
        cfg.runner['type'] = 'InfiniteEpochBasedRunner'
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and cfg.optimizer_config is not None \
            and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # currently only support single images testing
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        assert val_samples_per_gpu == 1, \
            'currently only support single images testing'

        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        
        if "Iter" in cfg.runner['type']:
            eval_cfg['by_epoch'] = False
        # eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'


        # Prepare `model_init` dataset for model initialization. In most cases,
        # the `model_init` dataset contains support images and few shot
        # annotations. The meta-learning based detectors will extract the
        # features from images and save them as part of model parameters.
        # The `model_init` dataset can be mutually configured or
        # randomly selected during runtime.
        if cfg.data.get('model_init', None) is not None:
            # The randomly selected few shot support during runtime can not be
            # configured offline. In such case, the copy datasets are designed
            # to directly copy the randomly generated support set for model
            # initialization. The copy datasets copy the `data_infos` by
            # passing it as argument and other arguments can be different
            # from training dataset.
            if cfg.data.model_init.pop('copy_from_train_dataset', False):
                if cfg.data.model_init.ann_cfg is not None:
                    warnings.warn(
                        'model_init dataset will copy support '
                        'dataset used for training and original '
                        'ann_cfg will be discarded', UserWarning)
                # modify dataset type to support copying data_infos operation
                cfg.data.model_init.type = \
                    get_copy_dataset_type(cfg.data.model_init.type)
                if not hasattr(dataset[0], 'get_support_data_infos'):
                    raise NotImplementedError(
                        f'`get_support_data_infos` is not implemented '
                        f'in {dataset[0].__class__.__name__}.')
                cfg.data.model_init.ann_cfg = [
                    dict(data_infos=dataset[0].get_support_data_infos())
                ]
            # The `model_init` dataset will be saved into checkpoint, which
            # allows model to be initialized with these data as default, if
            # the config of data is not be overwritten during testing.
            cfg.checkpoint_config.meta['model_init_ann_cfg'] = \
                cfg.data.model_init.ann_cfg
            samples_per_gpu = cfg.data.model_init.pop('samples_per_gpu', 1)
            workers_per_gpu = cfg.data.model_init.pop('workers_per_gpu', 1)
            model_init_dataset = build_dataset(cfg.data.model_init)
            # Noted that `dist` should be FALSE to make all the models on
            # different gpus get same data results in same initialized models.
            model_init_dataloader = build_dataloader(
                model_init_dataset,
                samples_per_gpu=samples_per_gpu,
                workers_per_gpu=workers_per_gpu,
                dist=False,
                shuffle=False)

            # eval hook for meta-learning based query-support detector, it
            # supports model initialization before regular evaluation.
            eval_hook = QuerySupportDistEvalHook \
                if distributed else QuerySupportEvalHook
            runner.register_hook(
                eval_hook(model_init_dataloader, val_dataloader, **eval_cfg),
                priority='LOW')
        else:
            # for the fine-tuned based methods, the evaluation is the
            # same as mmdet.
            eval_hook = DistEvalHook if distributed else EvalHook
            runner.register_hook(
                eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loader, cfg.workflow)
