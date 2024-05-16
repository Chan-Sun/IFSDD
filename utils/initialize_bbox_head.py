import argparse
import os

import torch
from mmcv.runner.utils import set_random_seed

# VOC config
NEU_TAR_SIZE = 6

def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--src1', type=str, help='Path to the main checkpoint')
    parser.add_argument(
        '--src2',
        type=str,
        default=None,
        help='Path to the secondary checkpoint. Only used when combining '
        'fc layers of two checkpoints')
    parser.add_argument(
        '--save-dir', type=str, default=None, help='Save directory')
    parser.add_argument(
        '--method',
        choices=['combine', 'remove', 'random_init'],
        required=True,
        help='Reshape method. combine = combine bbox heads from different '
        'checkpoints. remove = for fine-tuning on novel dataset, remove the '
        'final layer of the base detector. random_init = randomly initialize '
        'novel weights.')
    parser.add_argument(
        '--param-name',
        type=str,
        nargs='+',
        default=['roi_head.bbox_head.fc_cls', 'roi_head.bbox_head.fc_reg'],
        help='Target parameter names')
    parser.add_argument(
        '--tar-name',
        type=str,
        default='base_model',
        help='Name of the new checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    return parser.parse_args()


def random_init_checkpoint(param_name, is_weight, tar_size, checkpoint, args):
    """Either remove the final layer weights for fine-tuning on novel dataset
    or append randomly initialized weights for the novel classes.

    Note: The base detector for LVIS contains weights for all classes, but only
    the weights corresponding to base classes are updated during base training
    (this design choice has no particular reason). Thus, the random
    initialization step is not really necessary.
    """
    weight_name = param_name + ('.weight' if is_weight else '.bias')
    pretrained_weight = checkpoint['state_dict'][weight_name]
    prev_cls = pretrained_weight.size(0)
    if 'fc_cls' in param_name:
        prev_cls -= 1
    if is_weight:
        feat_size = pretrained_weight.size(1)
        new_weight = torch.rand((tar_size, feat_size))
        torch.nn.init.normal_(new_weight, 0, 0.01)
    else:
        new_weight = torch.zeros(tar_size)
    new_weight[:prev_cls] = pretrained_weight[:prev_cls]
    if 'fc_cls' in param_name:
        new_weight[-1] = pretrained_weight[-1]  # bg class
    checkpoint['state_dict'][weight_name] = new_weight


def combine_checkpoints(param_name, is_weight, tar_size, checkpoint,
                        checkpoint2, args):
    """Combine base detector with novel detector.

    Feature extractor weights are from the base detector. Only the final layer
    weights are combined.
    """
    if not is_weight and param_name + '.bias' not in checkpoint['state_dict']:
        return
    if not is_weight and param_name + '.bias' not in checkpoint2['state_dict']:
        return
    weight_name = param_name + ('.weight' if is_weight else '.bias')
    pretrained_weight = checkpoint['state_dict'][weight_name]
    prev_cls = pretrained_weight.size(0)
    if 'fc_cls' in param_name:
        prev_cls -= 1
    if is_weight:
        feat_size = pretrained_weight.size(1)
        new_weight = torch.rand((tar_size, feat_size))
    else:
        new_weight = torch.zeros(tar_size)
    new_weight[:prev_cls] = pretrained_weight[:prev_cls]

    checkpoint2_weight = checkpoint2['state_dict'][weight_name]

    if 'fc_cls' in param_name:
        new_weight[prev_cls:-1] = checkpoint2_weight[:-1]
        new_weight[-1] = pretrained_weight[-1]
    else:
        new_weight[prev_cls:] = checkpoint2_weight
    checkpoint['state_dict'][weight_name] = new_weight
    return checkpoint


def reset_checkpoint(checkpoint):
    if 'scheduler' in checkpoint:
        del checkpoint['scheduler']
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    if 'iteration' in checkpoint:
        checkpoint['iteration'] = 0


def main():
    args = parse_args()
    set_random_seed(args.seed)
    checkpoint = torch.load(args.src1)
    save_name = args.tar_name + f'_fine-tuning.pth'
    save_dir = args.save_dir \
        if args.save_dir != '' else os.path.dirname(args.src1)
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    reset_checkpoint(checkpoint)

    TAR_SIZE = NEU_TAR_SIZE

    if args.method == 'remove':
        # Remove parameters
        for param_name in args.param_name:
            del checkpoint['state_dict'][param_name + '.weight']
            if param_name + '.bias' in checkpoint['state_dict']:
                del checkpoint['state_dict'][param_name + '.bias']
    elif args.method == 'combine':
        checkpoint2 = torch.load(args.src2)
        tar_sizes = [TAR_SIZE + 1, TAR_SIZE * 4]
        for idx, (param_name,
                  tar_size) in enumerate(zip(args.param_name, tar_sizes)):
            combine_checkpoints(param_name, True, tar_size, checkpoint,
                                checkpoint2, args)
            combine_checkpoints(param_name, False, tar_size, checkpoint,
                                checkpoint2, args)
    elif args.method == 'random_init':
        tar_sizes = [TAR_SIZE + 1, TAR_SIZE * 4]
        for idx, (param_name,
                  tar_size) in enumerate(zip(args.param_name, tar_sizes)):
            random_init_checkpoint(param_name, True, tar_size, checkpoint,
                                   args)
            random_init_checkpoint(param_name, False, tar_size, checkpoint,
                                   args)
    else:
        raise ValueError(f'not support method: {args.method}')

    torch.save(checkpoint, save_path)
    print('save changed checkpoint to {}'.format(save_path))


if __name__ == '__main__':
    main()
