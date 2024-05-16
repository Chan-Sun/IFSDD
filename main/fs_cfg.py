import os.path as osp

def SetDefectDataset(cfg,
                    defect_data="NEU_DET",
                    setting ="SPLIT1_SEED1_10SHOT",
                    pretrain_weights=None):
    
    assert defect_data in ["NEU_DET","DeepPCB"]

    work_root = osp.dirname(__file__)
    data_root = f'{work_root}/../dataset/{defect_data}/'

    total_class_num = 6

    if "Pretrain" in setting:
    ### setting = Pretrain_SPLIT1
        class_num = int(total_class_num/2)
        split = setting.split("_")[-1][-1]
        cfg.data.val.classes=f'BASE_CLASSES_SPLIT{split}'
        cfg.data.test.classes=f'BASE_CLASSES_SPLIT{split}'        
        if cfg.data.train.type == "FewShotDefectDataset":
        ### for faster rcnn pretrain setting
            cfg.data.train.data_root = data_root
            cfg.data.train.classes=f'BASE_CLASSES_SPLIT{split}'
            cfg.data.train.img_prefix = f'images'
            cfg.data.train.defect_name = defect_data
            cfg.data.train.ann_cfg[0]["ann_file"]=f"annotations/trainval.json"
        else:
        ### for fsdetview, mpsr
            cfg.data.train.dataset.classes=f'BASE_CLASSES_SPLIT{split}'
            cfg.data.train.dataset.data_root = data_root
            cfg.data.train.dataset.defect_name = defect_data
            cfg.data.train.dataset.ann_cfg[0]["ann_file"]=f"annotations/trainval.json"
    else:
    ### setting = SPLIT1_SEED1_5SHOT
        shot = int(setting.split("_")[-1][:-4])
        split = int(setting.split("_")[0][-1])
        class_num = total_class_num

        cfg.data.val.classes=f'ALL_CLASSES_SPLIT{split}'
        cfg.data.test.classes=f'ALL_CLASSES_SPLIT{split}'
        

        if cfg.data.train.type == "FewShotDefectDefaultDataset":
            assert pretrain_weights is not None
            load_from_path = pretrain_weights
            if "algorithm" in cfg.keys():
                cfg.algorithm.architecture.model.init_cfg.checkpoint = load_from_path
                cfg.algorithm.distiller.teacher.init_cfg.checkpoint = load_from_path
            else:
                cfg.load_from = load_from_path

            cfg.data.train.data_root = data_root
            cfg.data.train.ann_cfg[0]["dataset_name"]=defect_data
            cfg.data.train.ann_cfg[0]["setting"]=setting
            cfg.data.train.num_novel_shots = shot
            cfg.data.train.num_base_shots = shot
            cfg.data.train.classes=f'ALL_CLASSES_SPLIT{split}'
            
    cfg.data.val.data_root = data_root
    cfg.data.val.defect_name = defect_data
    cfg.data.val.ann_cfg[0]["ann_file"]=f"annotations/test.json"
    cfg.data.test.data_root = cfg.data.val.data_root 
    cfg.data.test.defect_name = defect_data
    cfg.data.test.ann_cfg[0]["ann_file"]=f"annotations/test.json"
    cfg.model.roi_head.bbox_head.num_classes=class_num

    return cfg    