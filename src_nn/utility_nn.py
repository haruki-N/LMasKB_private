import random
import os
import argparse
import numpy as np
import torch


def get_args_cls():
    parser = argparse.ArgumentParser(description="1-to-N Relationships")
    parser.add_argument('--relation', type=int, default=2, help="number of relation")
    parser.add_argument('--father', type=int, default=36, help="number of children")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--mix_ratio', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--epoch_num', type=int, default=60, help="number of epochs for train")
    parser.add_argument('--batch_size', type=int, default=4, help="number of batch size")
    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--save_flag', action='store_true', help="set flag when you want to save models parameters")
    parser.add_argument('--gpu_number', default='0', help="which GPU to use")

    args = parser.parse_args()
    return args


def fix_seed(seed):
    """
        ランダムシードの固定
    :param seed: int
        シード値
    :return: None
    """
    # random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Tensorflow
    # tf.random.set_seed(seed)


def identify_outliers(y: list, mode='iqr', return_thresh=False):
    if mode == 'iqr':  # 四分位範囲を用いた外れ値検出
        quartile_1, quartile_3 = np.percentile(y, [25, 75])  # 第一四分位数, 第三四分位数を取得
        iqr = quartile_3 - quartile_1  # 四分位範囲の計算
        # lower_bound = quartile_1 - (iqr * 1.5)  # 下限
        upper_bound = quartile_3 + (iqr * 1.5)  # 上限
        # outlier_bool = ((y > upper_bound) | (y < lower_bound))
        outlier_bool = (y > upper_bound)
        outlier_index = np.nonzero(outlier_bool)[0]
        outliers = np.array(y)[outlier_bool].tolist()

        return outlier_index, outliers

    elif mode == 'std':
        # 標準偏差を用いた外れ値検出
        mean = np.mean(y)
        std = np.std(y)
        # upper_bound = mean + 3 * std
        upper_bound = mean + std
        outlier_bool = (y > upper_bound)
        outlier_index = np.nonzero(outlier_bool)[0]
        outliers = np.array(y)[outlier_bool].tolist()
        if return_thresh:
            return outlier_index, outliers, upper_bound

        return outlier_index, outliers

    else:
        print('Set iqr, std for "mode" argument.')


def get_indices(one_hot_tensor, value):
    dim = one_hot_tensor.dim()
    if dim == 1:
        return (one_hot_tensor == value).nonzero().view(-1)
    elif dim == 2:
        indices = list()
        for one_hot in one_hot_tensor:
            indices.append((one_hot == value).nonzero().view(-1).tolist())

        return torch.tensor(indices)
