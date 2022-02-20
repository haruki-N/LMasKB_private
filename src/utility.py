import torch
import numpy as np
import argparse
import random
import itertools
import math
import os


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


def float_to_str(f):
    float_string = repr(f)
    if 'e' in float_string:  # detect scientific notation
        digits, exp = float_string.split('e')
        digits = digits.replace('.', '').replace('-', '')
        exp = int(exp)
        zero_padding = '0' * (abs(int(exp)) - 1)  # minus 1 for decimal point in the sci notation
        sign = '-' if f < 0 else ''
        if exp > 0:
            float_string = '{}{}{}.0'.format(sign, digits, zero_padding)
        else:
            float_string = '{}0.{}{}'.format(sign, zero_padding, digits)
    return float_string


def get_args():
    parser = argparse.ArgumentParser(description="1-to-N Relationships")
    parser.add_argument('--father', type=int, default=360, help="number of children")
    parser.add_argument('--epoch_num', type=int, default=40, help="number of epochs for train")
    parser.add_argument('--batch_size', type=int, default=8, help="number of batch size")
    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--mix_ratio', type=int, default=0, choices=[0, 10, 30, 50, 70, 80])
    parser.add_argument('--save_flag', action='store_true', help="set flag when you want to save models parameters")
    parser.add_argument('--freeze_encoder', action='store_true',
                        help="set flag when you want to freeze encoder parematers for the proceeding task.")
    parser.add_argument('--second_task', default=None, choices=['mixing', 'fewshots'])
    parser.add_argument('--gpu_number', default='0', help="which GPU to use")

    args = parser.parse_args()
    return args


def get_args_cls():
    parser = argparse.ArgumentParser(description="1-to-N Relationships")
    parser.add_argument('--relation', type=int, default=2, help="number of relation")
    parser.add_argument('--father', type=int, default=36, help="number of children")
    parser.add_argument('--numbering', action='store_true')
    parser.add_argument('--mix_ratio', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--epoch_num', type=int, default=60, help="number of epochs for train")
    parser.add_argument('--batch_size', type=int, default=4, help="number of batch size")
    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--rand_init', action='store_true', default=False)
    parser.add_argument('--save_flag', action='store_true', help="set flag when you want to save models parameters")
    parser.add_argument('--freeze_encoder', action='store_true',
                        help="set flag when you want to freeze encoder parematers for the proceeding task.")
    parser.add_argument('--second_task', default=None, choices=['mixing', 'fewshots'])
    parser.add_argument('--gpu_number', default='0', help="which GPU to use")

    args = parser.parse_args()
    return args


def candidate_from_output(output, child_labels):
    """
    :param output:
        モデルが出力した確率分布
    :param child_labels:
        子どもの名前リスト
    :return:
    """
    idx = torch.argmax(output.cpu()).item()
    return child_labels[idx]


def flatten_list(li: list):
    """
        高次元リストを1次元リストに平坦化する
    :param li:
        1以上次元のリスト
    :return:
        平坦化されたリスト
    """
    for el in li:
        if isinstance(el, list):
            yield from flatten_list(el)
        else:
            yield el


def slice_list(lists: list, position=None) -> list:
    """
    リストをスライスする.

    :param lists: list
        スライス対象のリストの配列
    :param position:
        リストをスライスする位置idx. Noneでリストをスライスしない.
    :return:
        スライスされたリストorそのままのリストの配列を返す
    """
    if position:
        return [one_list[:position] for one_list in lists]
    else:
        return lists


def get_prob_top_children(model, device, encoded_data, numerical_settings):
    """
        予測確率分布のうちtop-Nコの確率値を取得し、子どものid番号(tokenized_id)順にソートして返す。

    :param model:
    :param device:
    :param encoded_data:
        モデルに入力するtokenized data（Questions）
    :param numerical_settings:
        実験に関する数値設定辞書
    :return:
        ソートされたtop-Nの確率を格納したリスト
        (N=1人の親に対する子どもの人数)
    """
    relation_num = numerical_settings['relation_num']
    if relation_num == 0:
        relation_num = 4
    probs = model.estimate(encoded_data, device, return_tensor=True)
    top_values, top_idx = torch.topk(probs, relation_num)
    top_idx, sort_idx = torch.sort(top_idx)
    sorted_values = torch.gather(top_values.cpu(), -1, sort_idx.cpu())

    return sorted_values


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


def random_mask_children(raw_sentences, child_pairs, randomize=True):
    """
        to prepare specific task: S has children named O_n <mask>
    :param raw_sentences: list
        raw_sentence should be like 'S has children named O_1, O_2, ...'
    :param child_pairs: list
        child_pair should be like 'O_1, O_2, ...'
    :return: list (containes two lists)
        mask sentences: ['S has children named O_n <mask>', ...]
        labels for mask: ['O_n-1, O_n+1, ...', 'O_n+1, O_n-1, ...', ...]
    """
    fix_seed(42)
    masked_sentences = []
    labels = []
    for sentence, child_pair in zip(raw_sentences, child_pairs):
        children = child_pair.split(' ')
        children = [child.strip(',.') for child in children]
        if randomize:
            poped_child = random.choice(children[1:])
            children.remove(poped_child)
            random.shuffle(children)   # 残りの子どもの順序をランダムにシャッフル
        else:
            poped_child = children[0]
            children.remove(poped_child)

        if 'named' in sentence:
            masked_sentence = sentence[:sentence.index('named')+6] + poped_child + ', <mask>'
        elif 'are' in sentence:
            masked_sentence = sentence[:sentence.index('are')+4] + poped_child + ', <mask>'
        label = ', '.join(children)

        masked_sentences.append(masked_sentence)
        labels.append(label)


    assert len(raw_sentences) == len(masked_sentences), 'Something wrong while processing'
    assert len(child_pairs) == len(labels), 'Something wrong while processing'

    return [masked_sentences, labels]


def get_permutations(masked_sent_list, children_list, first=None):
    """

    :param masked_sent_list:
        LMに入力するmask文のリスト.permutationの数にしたがってduplicateされる
    :param children_list:
        siblingsのリスト
    :param first:
        O_1を中心としてどのようにpermutationを取得するか
        各こどもが1回ずつO_1に割り当てられるような組み合わせが欲しい→'shuffle'
        O_1をあえて固定し、O_2以降のみをシャッフル→'fix'
        全組み合わせを取得→None
    :return:
        two lists
    """
    assert len(masked_sent_list) == len(children_list), "make sure that lengths are same on two lists"
    duplicated_masked_list = []
    children_all_permutations = []
    for masked_sent, children in zip(masked_sent_list, children_list):
        permutations = list(itertools.permutations(children))
        permutations = [list(perm) for perm in permutations]
        if first == 'shuffle':
            # O_1のみをシャッフルの対象と考える。全ての子どもがO_1に割り当てられるような組み合わせを一つずつ取得
            d = math.factorial(len(children)-1)
            permutations = [random.choice(permutations[d*i:d*(i+1)]) for i in range(len(children))]
        elif first == 'fix':
            # O_1は固定したまま、O_2以降のみをシャッフル
            d = math.factorial(len(children)-1)
            permutations = permutations[:d]

        num_permutation = len(permutations)
        duplicated_masked_list.extend([masked_sent]*num_permutation)
        children_all_permutations.extend(permutations)

    assert len(duplicated_masked_list) == len(children_all_permutations),\
        "something wrong while processing in get_permutation function"

    return duplicated_masked_list, children_all_permutations


def decompose_relation(parents: list, children: list):
    """

    :param parents:
        ['S1', 'S2', ...]
    :param children:
        [['C1', 'C2', ...], [,,,], ...]
    :return:
    """
    duplicated_sentences = []
    childs = []
    for parent, child_pair in zip(parents, children):
        num_children = len(child_pair)
        sentence = f'Who is the child of {parent}?'
        duplicated_sentences.extend([sentence] * num_children)
        childs.extend(child_pair)

    return duplicated_sentences, childs


def numbering_children(parents: list, children: list, decompose=True, mask=True):
    """

    :param parents:
        ['sentence1', 'sentence2', ...]
    :param children:
        [['C1', 'C2', ...], [...], ...]
    :param decompose:
        True → prepare one sentence per one single child
        Flase → prepare one sentence including all children
    :param mask:
        True → return masked sentences
        False → return non masked sentences
    :return:
    """
    new_inputs = []
    childs = []
    for parent, child_pair in zip(parents, children):
        num_children = len(child_pair)
        numbering = ['first', 'second', 'third', 'fourth']
        if decompose:
            if mask:
                new_inputs.extend([
                    f'The {i_th} child of {parent} is <mask>' for i_th in numbering[:num_children]
                ])
            else:
                new_inputs.extend([
                    f'Who is the {i_th} child of {parent}?' for i_th in numbering[:num_children]
                ])
            childs.extend(child_pair)

        else:
            concat = [f'The {i_th} child of {parent} is {child}.' for i_th, child in zip(numbering, child_pair)]
            if mask:
                masked = f'The children of {parent} are <mask>'
                new_inputs.append(' '.join(concat) + ' ' + masked)
            else:
                question = f'Who are the children of {parent}?'
                new_inputs.append(' '.join(concat) + ' ' + question)
            childs.append(' '.join(child_pair))

    return new_inputs, childs
