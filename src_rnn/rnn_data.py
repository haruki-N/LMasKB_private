import torch
import re
import pandas as pd
from rnn_utility import slice_list

SOS_token = 0
EOS_token = 1

def numbering_children(parents: list, children: list, decompose=False, mask=True):
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


def get_dataset(file_path: str, numerical_settings: dict, train=True, one_token=True,
                sentence='masked', option=None):
    """
    csvファイルからデータを読み込む.

    :param file_path: str
        ファイルパス
    :param numerical_settings: dict
        子どもやRelationの数などを格納した辞書.
    :param train : bool
        Trueで訓練用のデータ, Falseで評価データを取得.
    :param one_token: bool
        子どもの名前をone-token(1単語)のみとするかどうか.
        Trueでfirst name(正確には親と被っていない名前の部分)を取得. Falseでフルネーム.
    :param sentence: str
        返す文をmasked, raw, questionから選択
    :param option: str
        その他特定のカラムを取得したい時に指定
    :return:
        データのリストを複数返す.
    """
    df = pd.read_csv(file_path, encoding='utf-8')
    children_num = numerical_settings['children_num']
    relation_num = numerical_settings['relation_num']
    parent_num = children_num // relation_num
    assert children_num % relation_num == 0, \
        'Something wrong in number setting. Please check setting numbers(relation_num, etc.).'

    if train:
        if one_token:
            if option is not None:
                return df[option].tolist()[:children_num]

            if sentence == 'raw':
                return slice_list([df.childLabel1Token.tolist(), df.parentLabel.tolist(), df.sentence1Token.tolist()],
                                  children_num)
            if sentence == 'raw2':
                return slice_list([df.childLabel1Token.tolist(), df.parentLabel.tolist(), df.sentence1Token2.tolist()],
                                  children_num)
            elif sentence == 'question':
                return slice_list([df.childLabel1Token.tolist(), df.parentLabel.tolist(), df.question.tolist()],
                                  children_num)
            elif sentence == 'masked2':
                return slice_list([df.childLabel1Token.tolist(), df.parentLabel.tolist(), df.masked2.tolist()],
                                  children_num)

            return slice_list([df.childLabel1Token.tolist(), df.parentLabel.tolist(), df.masked.tolist()], children_num)
        else:
            if option is not None:
                return df[option].tolist()[:children_num]

            if sentence == 'raw':
                return slice_list([df.childLabel.tolist(), df.parentLabel.tolist(), df.sentence.tolist()],
                                  children_num)
            elif sentence == 'question':
                return slice_list([df.childLabel.tolist(), df.parentLabel.tolist(), df.question.tolist()],
                                  children_num)

            elif sentence == 'masked2':
                return slice_list([df.childLabel.tolist(), df.parentLabel.tolist(), df.masked2.tolist()],
                                  children_num)

            return slice_list([df.childLabel.tolist(), df.parentLabel.tolist(), df.masked.tolist()], children_num)

    else:  # mode = eval
        if one_token:
            if option is not None:
                return df[option].tolist()[:parent_num]

            if sentence == 'raw':
                return slice_list([[eval(ans) for ans in df.childLabel1Token.tolist()], df.sentence1Token.tolist()],
                                  parent_num)

            if sentence == 'raw2':
                return slice_list([[eval(ans) for ans in df.childLabel1Token.tolist()], df.sentence1Token2.tolist()],
                                  parent_num)

            elif sentence == 'question':
                return slice_list([[eval(ans) for ans in df.childLabel1Token.tolist()], df.question.tolist()],
                                  parent_num)

            elif sentence == 'masked2':
                return slice_list([[eval(ans) for ans in df.childLabel1Token.tolist()], df.masked2.tolist()],
                                  parent_num)

            return slice_list([[eval(ans) for ans in df.childLabel1Token.tolist()], df.masked.tolist()], parent_num)
        else:
            if option is not None:
                return df[option].tolist()[:parent_num]

            if sentence == 'raw':
                return slice_list([[eval(ans) for ans in df.childLabel.tolist()], df.sentence.tolist()], parent_num)

            elif sentence == 'question':
                return slice_list([[eval(ans) for ans in df.childLabel.tolist()], df.question.tolist()], parent_num)

            elif sentence == 'masked2':
                return slice_list([[eval(ans) for ans in df.childLabel.tolist()], df.masked2.tolist()], parent_num)

            return slice_list([df.childLabel.tolist(), df.masked.tolist()], parent_num)


def mix_dataset(train_files: list, eval_files: list, numerical_settings: dict, one_token=True,
                sentence='masked', option=None):
    children_num = numerical_settings['children_num']
    num_per_file = children_num // len(train_files)
    relation_nums = [int(re.findall(r"\d+", file_name)[-1]) for file_name in train_files]
    for i in relation_nums:
        assert num_per_file % i == 0, 'Numerical error ocurred in mix_dataset.'

    child_labels = []
    parent_labels = []
    masked_sentences = []
    eval_golds = []
    eval_questions = []
    options = []

    for train_file, eval_file, relation_num in zip(train_files, eval_files, relation_nums):
        settings = {
            'children_num': num_per_file,
            'relation_num': relation_num
        }
        if option is None:
            child_label, parent_label, masked = get_dataset(train_file, settings, train=True,
                                                            one_token=one_token, sentence=sentence, option=option)
            eval_gold, eval_question = get_dataset(eval_file, settings, train=False, one_token=one_token,
                                                   sentence=sentence, option=option)
            child_labels.extend(child_label)
            parent_labels.extend(parent_label)
            masked_sentences.extend(masked)
            eval_golds.extend(eval_gold)
            eval_questions.extend(eval_question)
        else:
            column = get_dataset(train_file, settings, train=True,
                                 one_token=one_token, sentence=sentence, option=option)
            options.extend(column)

    if option is not None:
        return options

    return child_labels, parent_labels, masked_sentences, \
           eval_golds, eval_questions


class MyDatasetGeneration(torch.utils.data.Dataset):
    def __init__(self, encoded_masked_text, encoded_label):
        self.encodings = encoded_masked_text
        self.answers = encoded_label['input_ids']

    def __getitem__(self, idx):
        input_ids = self.encodings['input_ids']
        attention_mask = self.encodings['attention_mask']

        return {
            'input_ids': input_ids[idx],
            'attention_mask': attention_mask[idx],
            'labels': self.answers[idx]
        }

    def __len__(self):
        return len(self.answers)


class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "<s>", 1: "</s>"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def read_vocab(src, tgt):
    print("Building vocaburaly...")

    input_lines = Vocab(src)
    output_lines = Vocab(tgt)

    return input_lines, output_lines


def prepare_data_for_metalearning(files, numerical_settings, show_ratio, src_tokenizer, tgt_tokenizer):
    child_pairs, parent_labels, _s_has_children_masked_sentences, _, _ = mix_dataset(files, files, numerical_settings,
                                                                                     sentence='masked', one_token=True)
    child_pairs = [eval(child_pair) for child_pair in child_pairs]
    # parentの名前から「,」「.」「'」を除外
    parent_labels = [parent_label.replace(',', '').replace('.', '').replace("'", '') for parent_label in parent_labels]

    masked_sentences = [f'Who are the children of {parent}?' for parent in parent_labels]
    split = len(masked_sentences) // 3
    one2two_child, one2two_parent, one2two_masked = child_pairs[:split], parent_labels[:split], masked_sentences[:split]
    print('ONE2TWO')
    print(one2two_child[:3])
    print(one2two_parent[:3])
    print(one2two_masked[:3])
    one2three_child, one2three_parent = child_pairs[split:2*split], parent_labels[split:2*split]
    one2three_masked = masked_sentences[split:2*split]
    one2four_child, one2four_parent = child_pairs[2*split:], parent_labels[2*split:]
    one2four_masked = masked_sentences[2*split:]
    assert len(one2two_masked) == len(one2three_masked) and len(one2three_masked) == len(one2four_masked),\
        'lengths are different from each other at prepare_data_for_metalearning'

    # split data into three sets
    threshold = len(one2two_masked) // 4
    train_child_pairs = one2two_child[:2*threshold] + one2three_child[:2*threshold] + one2four_child[:2*threshold]
    train_masked_sentences = one2two_masked[:2*threshold] + one2three_masked[:2*threshold] + one2four_masked[:2*threshold]

    meta_child_pairs = one2two_child[2*threshold:3*threshold] + one2three_child[2*threshold:3*threshold] + \
                       one2four_child[2*threshold:3*threshold]
    meta_masked_sentences = one2two_masked[2*threshold:3*threshold] + one2three_masked[2*threshold:3*threshold] + \
                            one2four_masked[2*threshold:3*threshold]

    test_child_pairs = one2two_child[3 * threshold:] + one2three_child[3 * threshold:] + one2four_child[3 * threshold:]
    test_masked_sentences = one2two_masked[3 * threshold:] + one2three_masked[3 * threshold:] + one2four_masked[3 * threshold:]

    # get new masked sentences: The {first, second, ...} child of S is <mask> (1to1)
    numbered_child_sentences, numbered_labels = numbering_children(parent_labels, child_pairs, decompose=True, mask=False)

    # extract seen relations from train set
    thresh = len(train_masked_sentences) // 3
    mix_ratio = (show_ratio // 10) * (thresh // 10)
    if mix_ratio == 0:
        mix_ratio = (show_ratio // 10) * thresh
    show_child = train_child_pairs[:mix_ratio] + train_child_pairs[thresh:thresh+mix_ratio] + train_child_pairs[thresh*2:thresh*2+mix_ratio]
    show_masked = train_masked_sentences[:mix_ratio] + train_masked_sentences[thresh:thresh+mix_ratio] + train_masked_sentences[thresh*2:thresh*2+mix_ratio]

    # prepare dataset
    # trainでは1to1は全て見せる
    show_child = [' '.join(child_pair) for child_pair in show_child]
    print(len(numbered_labels), len(numbered_child_sentences), len(show_child), len(show_masked))
    print(f'Train Example: 1to1[{numbered_child_sentences[0]} → {numbered_labels[0]}], 1toN[{show_masked[0]} → {show_child[0]}]')
    # mixing 1to1 and 1toN relations
    train_label_encoded = tgt_tokenizer.tokenize(numbered_labels+show_child, pad_length=8, check_oov=True)
    train_sentences_encoded = src_tokenizer.tokenize(numbered_child_sentences+show_masked, pad_length=24, check_oov=True)
    train_dataset = MyDatasetGeneration(train_sentences_encoded, train_label_encoded)

    meta_child_pairs = [' '.join(child_pair) for child_pair in meta_child_pairs]
    print(f'Meta Example: {meta_masked_sentences[0]} → {meta_child_pairs[0]}')
    meta_label_encoded = tgt_tokenizer.tokenize(meta_child_pairs, pad_length=8)
    meta_sentences_encoded = src_tokenizer.tokenize(meta_masked_sentences, pad_length=24)
    meta_dataset = MyDatasetGeneration(meta_sentences_encoded, meta_label_encoded)

    test_child_pairs = [' '.join(child_pair) for child_pair in test_child_pairs]
    print(f'Test Example: {test_masked_sentences[0]} → {test_child_pairs[0]}')
    test_label_encoded = tgt_tokenizer.tokenize(test_child_pairs, pad_length=8)
    test_sentences_encoded = src_tokenizer.tokenize(test_masked_sentences, pad_length=24)

    test_dataset = MyDatasetGeneration(test_sentences_encoded, test_label_encoded)

    return train_dataset, meta_dataset, test_dataset
