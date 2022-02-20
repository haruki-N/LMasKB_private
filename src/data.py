import numpy as np
import random
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utility import slice_list, numbering_children, decompose_relation, flatten_list


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_text, labels, eval=False):
        self.encodings = encoded_text
        self.labels = labels
        self.eval = eval

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


class MyDatasetGeneration(torch.utils.data.Dataset):
    def __init__(self, encoded_masked_text, encoded_label):
        self.encodings = encoded_masked_text
        self.answers = encoded_label['input_ids']

    def __getitem__(self, idx):
        input_ids = self.encodings.input_ids
        attention_mask = self.encodings.attention_mask

        return {
            'input_ids': input_ids[idx],
            'attention_mask': attention_mask[idx],
            'labels': self.answers[idx]
        }

    def __len__(self):
        return len(self.answers)


def prepare_dataset(input_sentences, labels, input_pad_length, label_pad_length, tokenizer, decoder_tokenizer=None):
    tokenized_inputs = tokenizer.batch_encode_plus(input_sentences, truncation=True, padding=True,
                                                   add_special_tokens=True, max_length=input_pad_length,
                                                   return_tensors='pt')
    if decoder_tokenizer:
        tokenized_labels = decoder_tokenizer.tokenize(labels, pad_length=label_pad_length)
    else:
        tokenized_labels = tokenizer.batch_encode_plus(labels, truncation=True, padding=True,
                                                       add_special_tokens=True, max_length=label_pad_length,
                                                       return_tensors='pt')
    dataset = MyDatasetGeneration(tokenized_inputs, tokenized_labels)

    return dataset


def prepare_dataloader(input_sentences, labels, input_pad_length, label_pad_length, batch_size,
                       tokenizer, decoder_tokenizer=None, shuffle=True):
    dataset = prepare_dataset(input_sentences, labels, input_pad_length, label_pad_length, tokenizer,
                              decoder_tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_loader


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


def split_train_valid_test(parents, children, train_ratio, valid_ratio, test_ratio):
    """
        1 to N relationsをtrain, valid, testに分けるだけ
    :param parents:
    :param children:
    :param train_ratio:
    :param valid_ratio:
    :param test_ratio:
    :return:
    """
    assert train_ratio + valid_ratio + test_ratio == 100, \
        'please balance the total of three ratios is equal to 100.'

    one2two = 0
    one2three = 0
    one2four = 0
    for child_pair in children:
        if len(child_pair) == 2:
            one2two += 1
        elif len(child_pair) == 3:
            one2three += 1
        elif len(child_pair) == 4:
            one2four += 1

    assert (one2two == one2three and one2two == one2four), 'number of relations are in dispropotion.'

    two_parent, two_children = parents[:one2two], children[:one2two]
    three_parent, three_children = parents[one2two: one2two+one2three], children[one2two: one2two+one2three]
    four_parent, four_children = parents[:-one2four], children[:-one2four]

    train_instances = int(len(two_parent) * (train_ratio / 100))
    valid_instances = int(len(two_parent) * (valid_ratio / 100))
    test_instances = int(len(two_parent) * (test_ratio / 100))

    train_parent = two_parent[:train_instances] + three_parent[:train_instances] + four_parent[:train_instances]
    valid_parent = two_parent[train_instances: train_instances+valid_instances] + \
                   three_parent[train_instances: train_instances+valid_instances] + \
                   four_parent[train_instances: train_instances+valid_instances]
    test_parent = two_parent[-test_instances:] + three_parent[-test_instances:] + four_parent[-test_instances:]

    train_chilren = two_children[:train_instances] + three_children[:train_instances] + four_children[:train_instances]
    valid_children = two_children[train_instances: train_instances + valid_instances] + \
                     three_children[train_instances: train_instances + valid_instances] + \
                     four_children[train_instances: train_instances + valid_instances]
    test_children = two_children[-test_instances:] + three_children[-test_instances:] + four_children[-test_instances:]

    print(f'#Father, train = {len(train_parent)}, valid = {len(valid_parent)}, test = {len(test_parent)}')
    train = (train_parent, train_chilren)
    valid = (valid_parent, valid_children)
    test = (test_parent, test_children)

    return train, valid, test


def prepare_data_for_metalearning(files, numerical_settings, show_ratio, batch_size, tokenizer, decoder_tokenizer=None):
    print(f'===== Dataset for meta learning will be prepared. (show_ratio = {show_ratio}) =====')
    child_pairs, all_parent_labels, _ = mix_dataset(files, numerical_settings, one_token=True)
    all_child_pairs = [eval(child_pair) for child_pair in child_pairs]
    print(f'#Fathers = {len(all_parent_labels)}, #Children = {len(list(flatten_list(all_child_pairs)))}')
    instances_for_each = numerical_settings['children_num'] // 3
    one2two_child = all_child_pairs[:instances_for_each]
    one2two_parent = all_parent_labels[:instances_for_each]
    one2two_masked = [f'The children of {parent} are <mask>' for parent in one2two_parent]

    one2three_child = all_child_pairs[instances_for_each: 2*instances_for_each]
    one2three_parent = all_parent_labels[instances_for_each: 2*instances_for_each]
    one2three_masked = [f'The children of {parent} are <mask>' for parent in one2three_parent]

    one2four_child = all_child_pairs[2*instances_for_each:]
    one2four_parent = all_parent_labels[2*instances_for_each:]
    one2four_masked = [f'The children of {parent} are <mask>' for parent in one2four_parent]

    # check if the splited realtions above is truly devided
    print(f'#instances: 1to2 = {len(one2two_parent)}, 1to3 = {len(one2three_parent)}, 1to4 = {len(one2four_parent)}')
    for relation, i in zip([one2two_child, one2three_child, one2four_child], [2, 3, 4]):
        for instance in relation:
            if len(instance) != i:
                raise RuntimeError(
                    f"the relations are not truly devided. check meta data setting.{instance}, {i}")

    # ---------- prepare 1to1 Relations -----------
    # get new masked sentences: The {first, second, ...} child of S is <mask> (1to1)
    one2one_sentences, one2one_label = numbering_children(all_parent_labels, all_child_pairs, decompose=True)

    # ---------- convert child_pair(['A', 'B', 'C']) into target sentence 'A B C'----------
    one2two_child_label = [' '.join(child_pair) for child_pair in one2two_child]
    one2three_child_label = [' '.join(child_pair) for child_pair in one2three_child]
    one2four_child_label = [' '.join(child_pair) for child_pair in one2four_child]

    # ---------- split 1toN data into three sets ----------
    show_instances = (numerical_settings['children_num'] * show_ratio) // 100
    threshold = show_instances // (2 * 3)   # 各relationでのfew_shot = show_instances // 3　であり、これをtrain, metaに二分
    print(f'#few_shot all instances in inner = {threshold} * 3 = {threshold * 3} (same as outer)')
    assert threshold != 0, 'the number of shown instances will be 0. please revise numerical settings.'

    # few shot that will be seen in inner training (with 1to1 relations)
    train_child_label = one2two_child_label[:threshold] + one2three_child_label[:threshold] + one2four_child_label[:threshold]
    train_masked_sentences = one2two_masked[:threshold] + one2three_masked[:threshold] + one2four_masked[:threshold]

    # few shot that will be seen in outer training (meta)
    meta_child_label = one2two_child_label[threshold: 2*threshold] + one2three_child_label[threshold: 2*threshold] + \
                       one2four_child_label[threshold: 2*threshold]
    meta_masked_sentences = one2two_masked[threshold: 2*threshold] + one2three_masked[threshold: 2*threshold] + \
                            one2four_masked[threshold: 2*threshold]

    # 1toN relations exluding few shot that will be seen in test phase
    # = checking how the model can answer correctly for unseen 1toN relations
    test_child_label = one2two_child_label[2*threshold:] + one2three_child_label[2*threshold:] + one2four_child_label[2*threshold:]
    test_masked_sentences = one2two_masked[2*threshold:] + one2three_masked[2*threshold:] + one2four_masked[2*threshold:]

    print(f'#1toN instances: inner = {len(train_masked_sentences)}, outer = {len(meta_masked_sentences)},'
          f'test = {len(test_masked_sentences)}')

    # prepare dataset
    # trainでは1to1は全て見せる
    print(f'Train Example: 1to1[{one2one_sentences[0]} → {one2one_label[0]}] #={len(one2one_sentences)}/ '
          f' 1toN[{train_masked_sentences[0]} → {train_child_label[0]}] #={len(train_masked_sentences)}')

    # mixing 1to1 and 1toN relations
    mixed_sentences = one2one_sentences+train_masked_sentences
    mixed_label = one2one_label+train_child_label
    if decoder_tokenizer is not None:
        train_loader = prepare_dataloader(train_masked_sentences, train_child_label,
                                          input_pad_length=24, label_pad_length=8, batch_size=batch_size,
                                          tokenizer=tokenizer, decoder_tokenizer=decoder_tokenizer, shuffle=True)
    else:
        train_loader = prepare_dataloader(train_masked_sentences, train_child_label,
                                          input_pad_length=24, label_pad_length=16, batch_size=batch_size,
                                          tokenizer=tokenizer, shuffle=True)

    print(f'Meta Example: {meta_masked_sentences[0]} → {meta_child_label[0]}')
    if decoder_tokenizer is not None:
        meta_loader = prepare_dataloader(meta_masked_sentences, meta_child_label, input_pad_length=24,
                                         label_pad_length=8, batch_size=batch_size,
                                         tokenizer=tokenizer, decoder_tokenizer=decoder_tokenizer, shuffle=True)
    else:
        meta_loader = prepare_dataloader(meta_masked_sentences, meta_child_label,
                                         input_pad_length=24, label_pad_length=16, batch_size=batch_size,
                                         tokenizer=tokenizer, shuffle=True)

    print(f'Test Example: {test_masked_sentences[0]} → {test_child_label[0]}')
    if decoder_tokenizer is not None:
        test_loader = prepare_dataloader(test_masked_sentences, test_child_label,
                                         input_pad_length=24, label_pad_length=8, batch_size=batch_size,
                                         tokenizer=tokenizer, decoder_tokenizer=decoder_tokenizer, shuffle=False)
    else:
        test_loader = prepare_dataloader(test_masked_sentences, test_child_label,
                                         input_pad_length=24, label_pad_length=16, batch_size=batch_size,
                                         tokenizer=tokenizer, shuffle=False)

    print('===== Dataset for meta learning is prepared =====')

    return train_loader, meta_loader, test_loader


def prepare_relations(parent_list, children_pairs, relation='child', relation_type='1toN', numbering=False):

    if relation_type == '1to1':
        # make "one" object for one subject
        input_sent = [f'Who is the {relation} of {parent}?' for parent in parent_list]
        answers = [relation + parent for parent in parent_list]

        return input_sent, answers

    elif relation_type == '1toN':
        # prepare N objects per one subject
        ordinals = ['first', 'second', 'third', 'fourth']
        one2one_input_sent = []
        one2one_answers = []
        one2N_input_sent = []
        object_pairs = []

        for parent, child_pair in zip(parent_list, children_pairs):
            if relation == 'child_decided':
                if numbering:
                    one2one_input_sent.extend([f'Who is the {ordinals[i]} child of {parent}?' for i, _ in enumerate(child_pair)])
                    objects = list(flatten_list(child_pair))
                else:
                    one2one_input_sent.extend([f'Who is the child of {parent}?' for _ in child_pair])
                    objects = list(flatten_list(child_pair))
                one2N_input_sent.append(f'Who are the children of {parent}?')
            else:   # 既に目的語が作成されているchild以外の場合は目的語を人工的に作成
                if numbering:
                    one2one_input_sent.extend(
                        [f'Who is the {ordinals[i]} {relation} of {parent}?' for i, _ in enumerate(child_pair)])
                    objects = [relation + parent + str(i + 1) for i, _ in enumerate(child_pair)]
                else:
                    one2one_input_sent.extend([f'Who is the {relation} of {parent}?' for i, _ in enumerate(child_pair)])
                    objects = [relation + parent + str(i + 1) for i, _ in enumerate(child_pair)]
                one2N_input_sent.append(f'Who are the {relation}s of {parent}?')

            one2one_answers.extend(objects)
            object_pairs.append(objects)
        return one2one_input_sent, one2one_answers, one2N_input_sent, object_pairs


def prepare_dataloader_cls_1to1(one2one_sent, one2one_answer, object2idx, tokenizer, batch_size, shuffle=True,
                                max_length=24):
    assert len(one2one_answer) == len(one2one_sent), \
        f'something wrong at adding new relations. #input:{len(one2one_sent)}, #output:{len(one2one_answer)}'

    tokenized_sent = tokenizer.batch_encode_plus(one2one_sent, truncation=True, padding=True,
                                                 max_length=max_length, add_special_tokens=True, return_tensors='pt')
    tgt_idx = torch.tensor([object2idx[obje] for obje in one2one_answer])

    print(f'#Total 1to1 instances = {len(one2one_answer)})')
    print('CLSIFFICATION 1to1 Ex.)')
    print(f'|INPUT: {one2one_sent[:2] + one2one_sent[-2:]}')
    print(f'|ANSWER: {one2one_answer[:2] + one2one_answer[-2:]}')
    print(f'|DATA_SIZE: input={len(one2one_sent)}, label={len(one2one_answer)}')
    print('---------------')
    dataset = MyDataset(tokenized_sent, tgt_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def prepare_dataloader_cls_1ton(parent_list, object_pairs, object2idx, num_all_objects,
                                tokenizer, batch_size, shuffle=True, max_length=24, relation='child'):
    if relation == 'child':
        input_sentences = [f"Who are the children of {parent}?" for parent in parent_list]
    elif relation == 'grandchild':
        input_sentences = [f'Who are the grandchildren of {parent}?' for parent in parent_list]
    elif relation == 'none':
        input_sentences = parent_list   # 既に入力文が作ってある場合
    else:   # 複数形が単にsをつけるだけの場合
        input_sentences = [f'Who are the {relation}s of {parent}?' for parent in parent_list]

    # object_pair_idx = [[object2idx[obje] for obje in object_pair] for object_pair in object_list]
    object_pair_idx = []
    for target in object_pairs:
        if isinstance(target, list):   # 1toN用のone-hot作成
            object_pair_idx.append([object2idx[obje] for obje in target])
        else:   # 1to1用のone-hot作成
            object_pair_idx.append([object2idx[target]])
    onehot_tgt = [torch.sum(torch.cat([F.one_hot(torch.tensor(obje_pair), num_classes=num_all_objects).view(1, -1)
                                       for obje_pair in object_pairs], dim=0),  dim=0)
                  for object_pairs in object_pair_idx]
    # print(onehot_tgt)
    tokenized_sent = tokenizer.batch_encode_plus(input_sentences, truncation=True, padding=True,
                                                 max_length=max_length, add_special_tokens=True, return_tensors='pt')
    print('CLSIFFICATION 1toN Ex.)')
    print(f'|INPUT: {input_sentences[-1]}')
    print(f'|ANSWER: {object_pairs[-1]}')
    print(f'|DATA_SIZE: input={len(input_sentences)}, label={len(onehot_tgt)}')
    print('---------------')

    dataset = MyDataset(tokenized_sent, onehot_tgt)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def prepare_dataloader_cls_mixing(parent_list, children_list, child2idx, num_all_children,
                                  mixing_ratio, tokenizer, batch_size, shuffle=True, numbering=True):
    # prepare 1to1 relatioins
    if numbering:
        one2one_sent, one2one_answer = numbering_children(parent_list, children_list, mask=False)
    else:
        one2one_sent, one2one_answer = decompose_relation(parent_list, children_list)

    children_idx = torch.tensor([child2idx[child] for child in one2one_answer])
    one2one_onehot_children = F.one_hot(children_idx, num_classes=num_all_children).tolist()
    one2one_onehot_children = [torch.tensor(i) for i in one2one_onehot_children]
    # prepare 1toN relaions
    one2two = 0
    one2three = 0
    one2four = 0
    for child_pair in children_list:
        if len(child_pair) == 2:
            one2two += 1
        elif len(child_pair) == 3:
            one2three += 1
        elif len(child_pair) == 4:
            one2four += 1
    print(f'#1toN: {one2two}, {one2three}, {one2four}')

    thresh = num_all_children // 3
    show_instances = int((mixing_ratio / 100) * thresh)

    one2n_sent = [f'Who are the children of {parent}?' for parent in parent_list]
    child_pair_idx = [[child2idx[child] for child in child_pair] for child_pair in children_list]
    one2n_onehot_children = [
        torch.sum(torch.cat([F.one_hot(torch.tensor(child_pair), num_classes=num_all_children).view(1, -1)
                             for child_pair in child_pairs], dim=0), dim=0)
        for child_pairs in child_pair_idx]
    one2two_sent, one2two_child = one2n_sent[:show_instances], one2n_onehot_children[:show_instances]
    one2three_sent = one2n_sent[one2two: one2two+show_instances]
    one2three_child = one2n_onehot_children[one2two: one2two+show_instances]
    one2four_sent = one2n_sent[-show_instances:]
    one2four_child = one2n_onehot_children[-show_instances:]

    mixed_sent = one2one_sent + one2two_sent + one2three_sent + one2four_sent
    mixed_label = one2one_onehot_children + one2two_child + one2three_child + one2four_child

    tokenized_sent = tokenizer.batch_encode_plus(mixed_sent, truncation=True, padding=True, max_length=24,
                                                 add_special_tokens=True, return_tensors='pt')
    dataset = MyDataset(tokenized_sent, mixed_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DataProcessor(object):
    def __init__(self, subject_list, child_pairs, relation_set, tokenizer, seed=42,
                 numbering=False, prepare_union=False):
        self.subjects = subject_list
        self.child_pairs = child_pairs
        self.relation_set = relation_set
        self.tokenizer = tokenizer
        self.seed = seed
        self.numbering = numbering
        self.prepare_union = prepare_union

        self.relation_datas = None
        self.subject_dict = None
        self.object_dict = None

    def __call__(self, batch_size):
        return self.prepare_data_loaders(batch_size)

    @staticmethod
    def _make_dict(classes):
        if len(classes) > len(set(classes)):
            category_dict = dict(zip(sorted(list(set(classes))), list(range(len(set(classes))))))
        else:
            category_dict = dict(zip(classes, list(range(len(classes)))))

        return category_dict

    def prepare_relation(self, relation='child'):
        # prepare N objects per one subject
        one2one_inputs = []
        one2one_answers = []

        one2N_inputs = []
        one2N_answers = []

        ordinals = ['first', 'second', 'third', 'fourth']

        for subject, child_pair in zip(self.subjects, self.child_pairs):
            if relation == 'child':
                if self.numbering:
                    one2one_inputs.extend(
                        [f'Who is the {ordinals[i]} child of {subject}?' for i, _ in enumerate(child_pair)])
                    objects = list(flatten_list(child_pair))
                else:
                    one2one_inputs.extend([f'Who is the child of {subject}?' for _ in child_pair])
                    objects = list(flatten_list(child_pair))
                one2N_inputs.append(f'Who are the children of {subject}?')
            else:  # 既に目的語が作成されているchild以外の場合は目的語を人工的に作成
                if self.numbering:
                    one2one_inputs.extend(
                        [f'Who is the {ordinals[i]} {relation} of {subject}?' for i, _ in enumerate(child_pair)])
                    objects = [relation + subject + str(i + 1) for i, _ in enumerate(child_pair)]
                else:
                    one2one_inputs.extend(
                        [f'Who is the {relation} of {subject}?' for i, _ in enumerate(child_pair)])
                    objects = [relation + subject + str(i + 1) for i, _ in enumerate(child_pair)]
                one2N_inputs.append(f'Who are the {relation}s of {subject}?')

            one2one_answers.extend(objects)
            one2N_answers.append(objects)

        data_dict = {
            '1to1input': one2one_inputs,
            '1to1answer': one2one_answers,
            '1toNinput': one2N_inputs,
            '1toNanswer': one2N_answers
        }

        return data_dict

    def prepare_data(self):
        relation_dicts = dict()
        for relation_type in self.relation_set:
            relation_dicts[relation_type] = self.prepare_relation(relation_type)

        self.relation_datas = relation_dicts
        return relation_dicts

    def prepare_dataset(self, inputs, answers, max_length=24, mode='1to1'):
        tokenized_input = self.tokenizer.batch_encode_plus(inputs, truncation=True, padding=True,
                                                           max_length=max_length, add_special_tokens=True,
                                                           return_tensors='pt')
        if mode == '1to1':
            target = torch.tensor([self.object_dict[obje] for obje in answers])
        elif mode == '1toN':
            target = [
                torch.sum(torch.cat([F.one_hot(torch.tensor(obje_pair), num_classes=len(self.object_dict)).view(1, -1)
                                     for obje_pair in object_pairs], dim=0), dim=0)
                for object_pairs in answers]

        dataset = MyDataset(tokenized_input, target)

        return dataset

    def prepare_data_loaders(self, batch_size):
        dataloaders_dict = dict()
        g = torch.Generator()
        g.manual_seed(self.seed)

        self.prepare_data()
        all_1to1_answer = list(flatten_list([self.relation_datas[rel]['1to1answer'] for rel in self.relation_set]))
        all_1to1_input = list(flatten_list([self.relation_datas[rel]['1to1input'] for rel in self.relation_set]))

        subject_dict = self._make_dict(self.subjects)
        self.subject_dict = subject_dict
        object_dict = self._make_dict(all_1to1_answer)
        self.object_dict = object_dict

        dataset = self.prepare_dataset(all_1to1_input, all_1to1_answer, mode='1to1')
        one2one_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                        generator=g, worker_init_fn=seed_worker)
        dataloaders_dict['1to1_all_relations'] = one2one_dataloader

        for relation in self.relation_set:
            target = self.relation_datas[relation]
            input_sentences = target['1toNinput']

            object_pair_idx = []
            for target_object in target['1toNanswer']:
                if isinstance(target_object, list):  # 1toN用のone-hot作成
                    object_pair_idx.append([self.object_dict[obje] for obje in target_object])
                else:  # 1to1用のone-hot作成
                    object_pair_idx.append([self.object_dict[target_object]])

            dataset = self.prepare_dataset(input_sentences, object_pair_idx, mode='1toN')
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                    generator=g, worker_init_fn=seed_worker)
            dataloaders_dict['1toN'+relation] = dataloader

        # if self.prepare_union:
        #     self.relation_datas['union'] = self.prepare_union_dataloader(['brother', 'sister'], return_dict=True)
        #     self.relation_set.append('union')

        # prepare all 1toN datas for training
        all_1toN_answer = list()
        for rel in self.relation_set:
            all_1toN_answer.extend(self.relation_datas[rel]['1toNanswer'])
        all_1toN_input = list(flatten_list([self.relation_datas[rel]['1toNinput'] for rel in self.relation_set]))

        object_pair_idx = []
        for target_object in all_1toN_answer:
            if isinstance(target_object, list):  # 1toN用のone-hot作成
                object_pair_idx.append([self.object_dict[obje] for obje in target_object])
            else:  # 1to1用のone-hot作成
                object_pair_idx.append([self.object_dict[target_object]])
        dataset = self.prepare_dataset(all_1toN_input, object_pair_idx, max_length=24, mode='1toN')
        one2N_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                      generator=g, worker_init_fn=seed_worker)
        dataloaders_dict['1toN_all_relations'] = one2N_dataloader
        # if self.prepare_union:
        #     self.relation_set.remove('union')

        # for evaluation (1to1)
        # eval_input = list(flatten_list([self.relation_datas[rel]['1toNinput'] for rel in self.relation_set]))
        eval_input = []
        eval_answer = []
        for rel in self.relation_set:
            eval_answer.extend(self.relation_datas[rel]['1toNanswer'])
            eval_input.extend([f'Who is the {rel} of {subject}?' for subject in self.subjects])

        print(f'input: {eval_input}')
        print(f'answer: {eval_answer}')
        object_pair_idx = []
        for target_object in eval_answer:
            if isinstance(target_object, list):  # 1toN用のone-hot作成
                object_pair_idx.append([self.object_dict[obje] for obje in target_object])
        dataset = self.prepare_dataset(eval_input, object_pair_idx, mode='1toN')
        dataloaders_dict['evaluation'] = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                    generator=g, worker_init_fn=seed_worker)

        return dataloaders_dict

    # def prepare_union_dataloader(self, union_set, return_dict=False, batch_size=8):
    #     assert set(union_set) not in set(self.relation_set),\
    #         f"make sure that {union_set} will be the subset of {self.relation_set}"
    #     union_dicts = [self.relation_datas[rel] for rel in union_set]
    #     union_data = dict()
    #
    #     union_data['1toNsubject'] = self.subjects
    #     union_data['1toNoperation'] = ['union' for _ in self.subjects]
    #     one2N_relation1 = union_dicts[0]['1toNrelation1']
    #     one2N_relation2 = union_dicts[1]['1toNrelation1']
    #     one2N_answers = []
    #     for idx in list(range(len(self.subjects))):
    #         union_answer = list(flatten_list([rel['1toNanswer'][idx] for rel in union_dicts]))
    #         one2N_answers.append(union_answer)
    #
    #     union_data['1toNrelation1'] = one2N_relation1
    #     union_data['1toNrelation2'] = one2N_relation2
    #     union_data['1toNanswer'] = one2N_answers
    #     if return_dict:
    #         return union_data
    #
    #     dataset = MyDataset(
    #         self._encode_onehot(union_data['1toNsubject'], my_dict=self.subject_dict),
    #         self._encode_onehot(union_data['1toNrelation1'], my_dict=self.relation_dict),
    #         self._encode_onehot(union_data['1toNrelation2'], my_dict=self.relation_dict),
    #         self._encode_onehot(union_data['1toNoperation'], my_dict=self.operation_dict),
    #         self._encode_multilabel_onehot(union_data['1toNanswer'], self.object_dict)
    #     )
    #
    #     g = torch.Generator()
    #     g.manual_seed(self.seed)
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
    #                             generator=g, worker_init_fn=seed_worker)
    #
    #     return dataloader
