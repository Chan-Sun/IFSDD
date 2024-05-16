time=$(date "+%Y%m%d%H%M")
work_folder=./work_dir/

### 1. Pretrain
# for split in 1 2 3
# do
# fs=Pretrain_SPLIT${split}
# for method in frcn_r101
# do
# for data in DeepPCB
# do
# python ./train.py \
#     ./config/base_train/${method}_base_training.py \
#     --gpu-ids 1 \
#     --work-dir ${work_folder}/${data}/${method}/${fs}/${time} \
#     --defect $data \
#     --fs_setting ${fs}
# done
# done
# done

### 2. Expand Box Dimension
# for method in frcn_r101
# do
# for split in 1 2 3
# do
# python ./utils/initialize_bbox_head.py \
#     --src1 ./weights/${method}_split${split}_base_training.pth \
#     --save-dir ./weights \
#     --method random_init \
#     --tar-name ${method}_split${split}
# done
# done

### 3. fine-tuning
# for method in iklf
# do
# for split in 1 2 3
# do
# for shot in 5 10 30
# do
# data=DeepPCB
# fs=SPLIT${split}_SEED1_${shot}SHOT
# python ./train.py \
#     ./config/fine_tune/${method}_fine_tuning.py \
#     --gpu-ids 0 \
#     --work-dir ${work_folder}/${data}/${method}/${fs}/${time} \
#     --defect $data \
#     --fs_setting ${fs}
# done
# done
# done

