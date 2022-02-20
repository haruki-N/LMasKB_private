import random
import copy
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def get_dataset(file_path: str, numerical_settings: dict, one_token=True):
    """
    csvファイルからデータを読み込む.

    :param file_path: str
        ファイルパス
    :param numerical_settings: dict
        子どもやRelationの数などを格納した辞書.
    :param one_token: bool
        子どもの名前をone-token(1単語)のみとするかどうか.
        Trueでfirst name(正確には親と被っていない名前の部分)を取得. Falseでフルネーム.
    :return:
        データのリストを複数返す.
    """
    df = pd.read_csv(file_path, encoding='utf-8')
    subject_num = numerical_settings['father_num']

    if one_token:
        return slice_list([df.childLabel1Token.tolist(), df.parentLabel.tolist(), df.masked.tolist()], subject_num)

    else:
        return slice_list([df.childLabel.tolist(), df.parentLabel.tolist(), df.masked.tolist()], subject_num)


def mix_dataset(train_files: list, numerical_settings: dict, one_token=True,):
    father_num = numerical_settings['father_num']
    num_per_file = father_num // len(train_files)

    child_labels = []
    parent_labels = []
    masked_sentences = []

    for train_file in train_files:
        # 各Relationについて処理
        settings = {
            'father_num': num_per_file
        }
        child_label, parent_label, masked = get_dataset(train_file, settings, one_token=one_token)
        child_labels.extend(child_label)
        parent_labels.extend(parent_label)
        masked_sentences.extend(masked)

    return child_labels, parent_labels, masked_sentences


def prepare_relation(parent_list, children_pairs, relation='child'):
    # prepare N objects per one subject
    one2one_subject = []
    one2one_relation = []
    one2one_operation = []
    one2one_answers = []

    one2N_subject = []
    one2N_relation = []
    one2N_operation = []
    one2N_answers = []

    for parent, child_pair in zip(parent_list, children_pairs):
        one2one_subject.extend([parent for _ in child_pair])
        one2one_relation.extend([relation for _ in child_pair])
        one2one_operation.extend(['element' for _ in child_pair])
        objects = [relation + parent + str(i + 1) for i, _ in enumerate(child_pair)]
        one2N_subject.append(parent)
        one2N_relation.append(relation)
        one2N_operation.append('set')

        one2one_answers.extend(objects)
        one2N_answers.append(objects)

    data_dict = {
        '1to1subject': one2one_subject,
        '1to1relation': one2one_relation,
        '1to1operation': one2one_operation,
        '1to1answer': one2one_answers,
        '1toNsubject': one2N_subject,
        '1toNrelation': one2N_relation,
        '1toNoperation': one2N_operation,
        '1toNanswer': one2N_answers
    }

    return data_dict


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, subject, relation1, relation2, operation, answer):
        self.subject = subject
        self.relation1 = relation1
        self.relation2 = relation2
        self.operation = operation
        self.answer = answer

    def __getitem__(self, idx):

        return {
            'subject': self.subject[idx],
            'relation1': self.relation1[idx],
            'relation2': self.relation2[idx],
            'operation': self.operation[idx],
            'answer': self.answer[idx]
        }

    def __len__(self):
        return len(self.subject)

    def get_example(self, idx):
        return {
            'subject': torch.argmax(self.subject[idx]),
            'relation1': torch.argmax(self.relation1[idx]),
            'relation2': torch.argmax(self.relation2[idx]),
            'operation': torch.argmax(self.operation[idx]),
            'answer': (self.answer[idx] == 1).nonzero(as_tuple=True)[0].tolist()
        }


class DataProcessor(object):
    def __init__(self, subject_list, child_pairs, relation_set, operations, union=False, seed=42):
        self.relation_dicts = None
        self.subject_dict = None
        self.object_dict = None
        self.relation_dict = None
        self.operation_dict = dict(zip(operations, list(range(len(operations)))))
        self.subjects = subject_list
        self.child_pairs = child_pairs
        self.relation_set = relation_set
        self.operations = operations
        self.union = union
        self.seed = seed

    def __call__(self, batch_size):
        return self.prepare_data_loaders(batch_size)

    def prepare_relation(self, relation='child'):
        # prepare N objects per one subject
        one2one_subject = []
        one2one_relation1 = []
        one2one_relation2 = []
        one2one_operation = []
        one2one_answers = []

        one2N_subject = []
        one2N_relation1 = []
        one2N_relation2 = []
        one2N_operation = []
        one2N_answers = []

        for parent, child_pair in zip(self.subjects, self.child_pairs):
            one2one_subject.extend([parent for _ in child_pair])
            one2one_relation1.extend([relation for _ in child_pair])
            one2one_relation2.extend(['none' for _ in child_pair])
            one2one_operation.extend(['element' for _ in child_pair])
            objects = [relation + parent + str(i + 1) for i, _ in enumerate(child_pair)]
            one2N_subject.append(parent)
            one2N_relation1.append(relation)
            one2N_relation2.append('none')
            one2N_operation.append('set')

            one2one_answers.extend(objects)
            one2N_answers.append(objects)

        data_dict = {
            '1to1subject': one2one_subject,
            '1to1relation1': one2one_relation1,
            '1to1relation2': one2one_relation2,
            '1to1operation': one2one_operation,
            '1to1answer': one2one_answers,
            '1toNsubject': one2N_subject,
            '1toNrelation1': one2N_relation1,
            '1toNrelation2': one2N_relation2,
            '1toNoperation': one2N_operation,
            '1toNanswer': one2N_answers
        }

        return data_dict

    @staticmethod
    def _encode_onehot(classes, vec_length=None, my_dict=None, return_dict=False):
        if my_dict:
            category_dict = my_dict
        else:
            if len(classes) > len(set(classes)):
                category_dict = dict(zip(sorted(list(set(classes))), list(range(len(set(classes))))))
            else:
                category_dict = dict(zip(classes, list(range(len(classes)))))
        if vec_length is None:
            vec_length = len(category_dict)
        category_idx = torch.tensor([category_dict[i] for i in classes])
        one_hot = F.one_hot(category_idx, num_classes=vec_length)

        if return_dict:
            return one_hot, category_dict
        return one_hot

    @staticmethod
    def _encode_multilabel_onehot(object_pairs, object_dicts):
        multi_labels = list()
        for object_pair in object_pairs:
            obj_idx = torch.Tensor([object_dicts[i] for i in object_pair]).long()
            _onehots = F.one_hot(obj_idx, num_classes=len(object_dicts))
            target = _onehots.sum(dim=0).int().tolist()
            multi_labels.append(target)
        return torch.Tensor(multi_labels)

    def prepare_data(self):
        relation_dicts = dict()
        for relation_type in self.relation_set:
            relation_dicts[relation_type] = self.prepare_relation(relation_type)

        self.relation_dicts = relation_dicts
        return relation_dicts

    def prepare_data_loaders(self, batch_size):
        dataloaders_dict = dict()
        g = torch.Generator()
        g.manual_seed(self.seed)

        self.prepare_data()
        all_1to1_operation = list(flatten_list([self.relation_dicts[rel]['1to1operation'] for rel in self.relation_set]))
        all_1to1_answer = list(flatten_list([self.relation_dicts[rel]['1to1answer'] for rel in self.relation_set]))
        all_1to1_subject = list(flatten_list([self.relation_dicts[rel]['1to1subject'] for rel in self.relation_set]))
        all_1to1_relation1 = list(
            flatten_list([self.relation_dicts[rel]['1to1relation1'] for rel in self.relation_set]))
        all_1to1_relation2 = list(
            flatten_list([self.relation_dicts[rel]['1to1relation2'] for rel in self.relation_set]))

        _, subject_dict = self._encode_onehot(all_1to1_subject, return_dict=True)
        self.subject_dict = subject_dict
        _, object_dict = self._encode_onehot(all_1to1_answer, return_dict=True)
        self.object_dict = object_dict
        all_rels = set(all_1to1_relation1)
        all_rels.add('none')
        all_rels = sorted(list(all_rels))
        _, relation_dict = self._encode_onehot(all_rels, return_dict=True)
        self.relation_dict = relation_dict

        dataset = MyDataset(self._encode_onehot(all_1to1_subject, my_dict=self.subject_dict),
                            self._encode_onehot(all_1to1_relation1, my_dict=self.relation_dict),
                            self._encode_onehot(all_1to1_relation2, my_dict=self.relation_dict),
                            self._encode_onehot(all_1to1_operation, my_dict=self.operation_dict),
                            self._encode_onehot(all_1to1_answer, my_dict=self.object_dict)
                            )
        one2one_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                        generator=g, worker_init_fn=seed_worker)
        dataloaders_dict['1to1_all_relations'] = one2one_dataloader

        for relation in self.relation_set:
            target = self.relation_dicts[relation]
            dataset = MyDataset(
                self._encode_onehot(target['1toNsubject'], my_dict=self.subject_dict),
                self._encode_onehot(target['1toNrelation1'], my_dict=self.relation_dict),
                self._encode_onehot(target['1toNrelation2'], my_dict=self.relation_dict),
                self._encode_onehot(target['1toNoperation'], my_dict=self.operation_dict),
                self._encode_multilabel_onehot(target['1toNanswer'], self.object_dict)
            )
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                    generator=g, worker_init_fn=seed_worker)
            dataloaders_dict['1toN'+relation] = dataloader

        if self.union:
            self.relation_dicts['union'] = self.prepare_union_dataloader(['brother', 'sister'], return_dict=True)
            self.relation_set.append('union')

        all_1toN_operation = list(
            flatten_list([self.relation_dicts[rel]['1toNoperation'] for rel in self.relation_set]))
        all_1toN_answer = list()
        for rel in self.relation_set:
            all_1toN_answer.extend(self.relation_dicts[rel]['1toNanswer'])
        all_1toN_subject = list(flatten_list([self.relation_dicts[rel]['1toNsubject'] for rel in self.relation_set]))
        all_1toN_relation1 = list(flatten_list([self.relation_dicts[rel]['1toNrelation1'] for rel in self.relation_set]))
        all_1toN_relation2 = list(flatten_list([self.relation_dicts[rel]['1toNrelation2'] for rel in self.relation_set]))

        dataset = MyDataset(self._encode_onehot(all_1toN_subject, my_dict=self.subject_dict),
                            self._encode_onehot(all_1toN_relation1, my_dict=self.relation_dict),
                            self._encode_onehot(all_1toN_relation2, my_dict=self.relation_dict),
                            self._encode_onehot(all_1toN_operation, my_dict=self.operation_dict),
                            self._encode_multilabel_onehot(all_1toN_answer, self.object_dict)
                            )
        one2N_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                      generator=g, worker_init_fn=seed_worker)
        dataloaders_dict['1toN_all_relations'] = one2N_dataloader
        if self.union:
            self.relation_set.remove('union')

        # for eval
        eval_subjct = list(flatten_list([self.relation_dicts[rel]['1toNsubject'] for rel in self.relation_set]))
        eval_relation1 = list(flatten_list([self.relation_dicts[rel]['1toNrelation1'] for rel in self.relation_set]))
        eval_relation2 = list(flatten_list([self.relation_dicts[rel]['1toNrelation2'] for rel in self.relation_set]))
        eval_operation = ['element' for _ in eval_subjct]
        eval_answer = []
        for rel in self.relation_set:
            eval_answer.extend(self.relation_dicts[rel]['1toNanswer'])
        dataset = MyDataset(
            self._encode_onehot(eval_subjct, my_dict=subject_dict),
            self._encode_onehot(eval_relation1, my_dict=self.relation_dict),
            self._encode_onehot(eval_relation2, my_dict=self.relation_dict),
            self._encode_onehot(eval_operation, my_dict=self.operation_dict),
            self._encode_multilabel_onehot(eval_answer, self.object_dict)
        )
        dataloaders_dict['evaluation'] = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                    generator=g, worker_init_fn=seed_worker)

        return dataloaders_dict

    def prepare_union_dataloader(self, union_set, return_dict=False, batch_size=8):
        assert set(union_set) not in set(self.relation_set),\
            f"make sure that {union_set} will be the subset of {self.relation_set}"
        union_dicts = [self.relation_dicts[rel] for rel in union_set]
        union_data = dict()

        union_data['1toNsubject'] = self.subjects
        union_data['1toNoperation'] = ['union' for _ in self.subjects]
        one2N_relation1 = union_dicts[0]['1toNrelation1']
        one2N_relation2 = union_dicts[1]['1toNrelation1']
        one2N_answers = []
        for idx in list(range(len(self.subjects))):
            union_answer = list(flatten_list([rel['1toNanswer'][idx] for rel in union_dicts]))
            one2N_answers.append(union_answer)

        union_data['1toNrelation1'] = one2N_relation1
        union_data['1toNrelation2'] = one2N_relation2
        union_data['1toNanswer'] = one2N_answers
        if return_dict:
            return union_data

        dataset = MyDataset(
            self._encode_onehot(union_data['1toNsubject'], my_dict=self.subject_dict),
            self._encode_onehot(union_data['1toNrelation1'], my_dict=self.relation_dict),
            self._encode_onehot(union_data['1toNrelation2'], my_dict=self.relation_dict),
            self._encode_onehot(union_data['1toNoperation'], my_dict=self.operation_dict),
            self._encode_multilabel_onehot(union_data['1toNanswer'], self.object_dict)
        )

        g = torch.Generator()
        g.manual_seed(self.seed)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                generator=g, worker_init_fn=seed_worker)

        return dataloader
