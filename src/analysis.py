import torch
from utility import flatten_list, get_prob_top_children, identify_outliers
from pprint import pprint
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


mpl.use('Agg')


def plot_single_fig(values, file_name, title, labels, x_label, y_label):
    plt.figure(figsize=(10, 5))
    if isinstance(values[0], list):
        for value, label in zip(values, labels):
            epoch_num = list(range(1, len(value) + 1))
            plt.plot(epoch_num, value, label=label)
    else:
        epoch_num = list(range(1, len(values) + 1))
        plt.plot(epoch_num, values, label=labels)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    # plt.grid(linestyle='dotted')

    plt.tight_layout()
    plt.savefig(file_name)
    print('Recording single fig. done')

# -------------- Acc & Loss -----------------

def show_acc_loss(numerical_settings: dict, train_acc: list, train_loss: list, file_name: str, labels=None):
    relation_num, children_num = numerical_settings['relation_num'], numerical_settings['father_num']
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    if isinstance(train_loss[0], list):
        loss_labels = ['Train loss', 'Valid loss']
        for loss, label in zip(train_loss, loss_labels):
            epoch_num = list(range(1, len(loss) + 1))
            plt.plot(epoch_num, loss, label=label)
    else:
        epoch_num = list(range(1, len(train_loss) + 1))
        plt.plot(epoch_num, train_loss, label='Train Loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title(f'#fathers={children_num}: Rel. memorization Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    if isinstance(train_acc[0], list):
        if labels:
            label = labels
        else:
            label = ['task1', 'task2', 'task3']
        for acc, label in zip(train_acc, label):
            epoch_num = list(range(1, len(acc) + 1))
            plt.plot(epoch_num, acc, label=label)
    else:
        epoch_num = list(range(1, len(train_acc) + 1))
        plt.plot(epoch_num, train_acc, label='Train Acc')
    plt.xlabel('epoch')
    plt.ylabel('Acc')
    plt.title(f'#fathers={children_num}: Rel. memorization Acc.')
    plt.legend()
    # plt.grid(linestyle='dotted')

    plt.tight_layout()
    plt.savefig(file_name)
    print('Recording Acc & Loss done')


# -------------- 最終的な確率分布全体を見るための関数 -----------------
def plt_local_hole(local_x, local_y, local_ticks, hole_x, hole_y, hole_ticks, title, file_name):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 2)
    plt.plot(local_x, local_y, marker='.')
    plt.title(f'Part of probability distribution')
    plt.xlabel('objects')
    plt.xticks(local_x, local_ticks)
    plt.xticks(rotation=30)
    plt.ylabel('Probability')
    plt.legend()

    plt.subplot(1, 2, 1)
    plt.plot(hole_x, hole_y, marker='.')
    plt.title(title)
    plt.xlabel('objects')
    plt.xticks(hole_x, hole_ticks)
    plt.ylabel('Probability')
    plt.legend()

    plt.tight_layout()

    plt.savefig(file_name)


def show_prob_distribution_classification(numerical_settings: dict, prob_dist: list, ans: list,
                                          title=None, file_name=None):
    children_num = numerical_settings['father_num']
    hole_ticks = [''] * len(prob_dist)
    hole_x = list(range(len(prob_dist)))

    outlier_idx, top_N, thresh = identify_outliers(prob_dist, mode='std', return_thresh=True)
    prob_one_zero = [1 if i > thresh else 0 for i in prob_dist]

    if title is None:
        title = f'prob dist {children_num}_1to1'
    if file_name is None:
        file_name = f"../1toN_result/RoBERTa_prob_1to1.png"

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(hole_x, prob_dist, marker='.')
    plt.title(title)
    plt.xlabel('objects')
    plt.xticks(hole_x, hole_ticks)
    plt.ylabel('Probability')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hole_x, prob_one_zero, marker='.')
    plt.title(f'1 or 0 distribution')
    plt.xlabel('objects')
    plt.xticks(hole_x, hole_ticks)
    plt.xticks(rotation=30)
    plt.ylabel('1 will be in the set')
    plt.legend()

    plt.tight_layout()

    plt.savefig(file_name)
    print('Recording Probability distribution done')


def show_prob_distribution_mask_prediction(numerical_settings: dict, prob_dist: list, ans_ids: list):
    relation_num, children_num = numerical_settings['relation_num'], numerical_settings['father_num']

    hole_ticks = [''] * len(prob_dist)  # ticksとしてデフォルトは空文字
    hole_x = list(range(len(prob_dist)))
    local_prob_dist = prob_dist[min(ans_ids) - 3:max(ans_ids) + 4]  # 正解のidの範囲を取得
    local_ticks = [''] * len(local_prob_dist)
    local_x = list(range(len(local_prob_dist)))
    for n, i in enumerate(sorted(ans_ids)):  # 正解の選択肢に対してtickとしてラベル付
        idx = i - min(ans_ids) + 3
        local_ticks[idx] = f'Ans{n + 1}'
        hole_ticks[i - 1] = f'{n + 1}'
    # Local
    plt_local_hole(numerical_settings,
                   local_x, local_prob_dist, local_ticks,
                   hole_x, prob_dist, hole_ticks,
                   f'../result/BART_prob-{children_num}-1to{relation_num}.png')
    print('Recording probability distribution done')


# -------------- エポック毎の確率分布の違いを追うための関数 -----------------
def record_epoch(epoch_number, model, device, encoded_data, numerical_settings=None):
    """
        あるエポックで確率分布を記録したリストを返す
    :param epoch_number: int
        記録タイミングのエポック
    :param model:
        モデル
    :param device:
        データを渡すdevice
    :param encoded_data:
        エンコーディング済みの質問文データ。
        複数sequenceを入力した場合は、各sequence毎の確率分布を均したリストを返す。
    :param numerical_settings:
        複数sequence渡す場合に、子どもの人数を把握するために使用する辞書。
    :return:
    """
    model.eval()
    with torch.no_grad():
        sequence_num = encoded_data['input_ids'].size()[0]
        sorted_values = get_prob_top_children(model, device, encoded_data, numerical_settings)
        if sequence_num == 1:
            output = sorted_values.tolist()
            return {'epoch_number': epoch_number, 'prob_dist': output}

        else:
            output = sum(sorted_values) / sequence_num
            return {'epoch_number': epoch_number, 'prob_dist': output.tolist()}


def show_avg_prob(model, device, encoded_data_list, file_name, numerical_settings=None):
    plt.figure(figsize=(15, 10))
    model.eval()
    x_s = np.arange(4)
    width = 0.2
    with torch.no_grad():
        # relationごとに処理する
        for i, encoded_data in enumerate(encoded_data_list):
            numerical_settings['relation_num'] = i+2
            sorted_values = get_prob_top_children(model, device, encoded_data, numerical_settings)
            avg_values = torch.mean(sorted_values, 0).tolist()
            if len(avg_values) < 4:   # padding to length 4
                for j in range(4-len(avg_values)):
                    avg_values.append(.0)
            plt.bar(x_s+width*i, avg_values, width=0.2, label=f'{i+2}Children')

    # plt.bar(probs_overall, label=labels, stacked=False)
    plt.title('Averaged probablity 2~4 children')
    plt.xlabel('Vocabulary')
    plt.ylabel('Probability')
    plt.xticks(x_s+width, ['child1', 'child2', 'child3', 'child4'])
    plt.legend()
    plt.grid(axis='y', linestyle='dotted')
    plt.savefig(file_name)
    print('Recording averaged porb. done')


def show_prob_epochs(prob_list: list, file_name: str, ans_names=None):
    """
        エポック毎の確率分布のグラフを出力
    :param prob_list:
        エポック毎の確率分布を格納したリスト
    :param file_name:
        プロット結果を保存するファイル名（ファイルパス）
    :param ans_names: list
        プロットの際に子どもの名前を表示したい時に渡す正解の子供の名前リスト。
    :return:
        None
    """
    if isinstance(prob_list, dict):
        item_count = len(prob_list['prob_dist'])
    else:
        item_count = len(prob_list[0]['prob_dist'])
    plt.figure(figsize=(10, 5))
    if ans_names:
        print(ans_names)
        ticks = [name for name in ans_names]
    else:
        ticks = [f'ans-{i}' for i in range(1, item_count + 1)]
    markers = ['.', 's', 'x', 'v', 'd', 'h', 'p', '+', 'o', '*', '|', '>', '<', '^', '_',
               '1', '2', '3', '4', '8', 'D', 'H'][:len(prob_list)]
    cm = plt.get_cmap("Blues")
    x_s = list(range(len(ticks)))
    for i, content in enumerate(zip(markers, prob_list), 1):
        marker, prob_dict = content
        epoch, prob = prob_dict['epoch_number'], prob_dict['prob_dist']
        prob = list(flatten_list(prob))  # リストを1次元にする
        print(prob)

        plt.plot(x_s, prob, label=f'epoch {epoch + 1}', marker=marker, color=cm(i / len(prob_list)))

    plt.title('Probablity changes thorough epochs')
    plt.xlabel('Vocabulary')
    plt.ylabel('Probability')
    plt.xticks(x_s, ticks)
    plt.legend()
    plt.grid(axis='y', linestyle='dotted')
    plt.savefig(file_name)
    print('Recording porb. through epoch done')


# --------- 出力確率が1/Nとどれほど差があるかを測定する ------------
def show_diff(model, device, data_loader, numerical_settings, file_name,
              child_label=None, tokenizer=None, diff=True, verbose=False):
    relation_num = numerical_settings['relation_num']
    model.eval()
    plt.figure(figsize=(15, 5))
    if relation_num == 0:   # Nがミックスされている場合
        split = 4
        top_Ns = []
        two_s = []
        three_s = []
        four_s = []
        others = []
        others_relations = {}
        with torch.no_grad():
            for batch in data_loader:
                probs = model.estimate(batch, device, return_tensor=True)
                for i, prob in enumerate(probs):
                    prob = prob.tolist()
                    outlier_idx, top_N = identify_outliers(prob, mode='std')
                    if len(top_N) == 2:
                        two_s.extend(top_N)
                    elif len(top_N) == 3:
                        three_s.extend(top_N)
                    elif len(top_N) == 4:
                        four_s.extend(top_N)
                    else:
                        others.extend(top_N)
                        if verbose:
                            if tokenizer:
                                parent = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
                                print(parent)
                                parent = ''.join(parent[1:parent.index('has')])
                                print(f'TYPE: {type(model)}')
                                if 'models.MyBART' in str(type(model)):
                                    children = tokenizer.convert_ids_to_tokens(outlier_idx)
                                else:
                                    children = [child_label[idx] for idx in outlier_idx]

                                others_relations[parent] = children

                    top_Ns.extend(top_N)

            print('Out of OUTLIERS(in show_diff) other than 2~4 rel.')
            pprint(others_relations)
            std = str(np.std(top_Ns))
            plt.subplot(1, 2, 1)
            labels = [f'2children({len(two_s)})', f'3children({len(three_s)})', f'4children({len(four_s)})', f'others({len(others)})']
            plt.hist([two_s, three_s, four_s, others], label=labels, stacked=True, bins=30)
            plt.xlabel('Probability')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(linestyle='dotted')
            plt.title('Probability  histgram(2~4Relations mixed)')

    else:   # Nが一意の場合
        split = relation_num
        top_Ns = []
        with torch.no_grad():
            for batch in data_loader:
                probs = model.estimate(batch, device, return_tensor=False)
                # tokenized_id順にソートされたtop-Nの確率値を取得
                top_Ns.extend(get_prob_top_children(model, device, batch, numerical_settings))

        if diff:   # 1/Nとの差分をとる
            criteria = 1 / relation_num
            top_Ns -= criteria

        std = str(torch.std(top_Ns.reshape(-1)).item())[:6]  # 標準偏差の取得
        top_Ns = top_Ns.reshape(-1).tolist()

        abs_max = max(abs(min(top_Ns)), max(top_Ns))
        plt.subplot(1, 2, 1)
        plt.hist(top_Ns, range=(-abs_max - 0.05, abs_max + 0.05), bins=200 // split,
                 rwidth=0.8, )
        plt.xlabel(f'Difference(%) [std = {std}]')
        plt.ylabel('Count')
        plt.grid(linestyle='dotted')
        plt.title(f'Probability distribution')

    plt.subplot(1, 2, 2)
    plt.boxplot(top_Ns, vert=False, showmeans=True)
    plt.xlabel(f'Difference(%) [std = {std}]')
    plt.title(f'Box plot of porbability distribution')
    plt.grid(axis='y', linestyle='dotted')
    plt.savefig(file_name)
    print('Recording Differences from 1/N done')

    if verbose:   # 外れ値がどんなものか見る
        if child_label:
            outlier_index, outliers_diff = identify_outliers(top_Ns, mode='iqr')
            print(outliers_diff)
            print(outlier_index)
            print(f'Len: {len(outliers_diff)}, Max:{max(outliers_diff)}, Min: {min(outliers_diff)}')
            outlier_children = [child_label[i] for i in outlier_index]
            if tokenizer:
                print('Analysis diffs. for BART')
                for i, idx in enumerate(outlier_index):
                    group_idx = idx - (idx % relation_num)
                    outlier_children_names = [child_label[group_idx + j] for j in range(relation_num)]
                    outlier_children_values = [top_Ns[group_idx + j] for j in range(relation_num)]
                    outlier_idx = outlier_children_values.index(outliers_diff[i])

                    # 取得した確率値はid番号順(＝予測番号順)となっていたので、子どもの名前をid番号順に並べ替える。
                    tokenized_ids = tokenizer.convert_tokens_to_ids(outlier_children_names)
                    sorter = zip(tokenized_ids, outlier_children_names)
                    _, sorted_names = zip(*sorter)
                    print(f'OUTLIER {sorted_names[outlier_idx]}: {outliers_diff[i]}')
                    print('CHILDREN:', end=' ')
                    print(*[pair for pair in
                            zip(sorted_names[:len(tokenized_ids)], outlier_children_values[:len(tokenized_ids)])],
                          end='\n\n')

            else:
                print('Analysis diff. for RoBERTa')
                for i, idx in enumerate(outlier_index):
                    group_idx = idx - (idx % relation_num)
                    outlier_children_names = [child_label[group_idx + j] for j in range(relation_num)]
                    outlier_children_values = [top_Ns[group_idx + j] for j in range(relation_num)]
                    outlier_idx = outlier_children_values.index(outliers_diff[i])
                    print(f'OUTLIER {outlier_children_names[outlier_idx]}: {outliers_diff[i]}')
                    print('CHILDREN:', end=' ')
                    print(*[pair for pair in zip(outlier_children_names, outlier_children_values)], end='\n\n')

