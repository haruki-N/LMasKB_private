import torch
import tqdm
import sys
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import BartTokenizer

from analysis import show_acc_loss
from data import MyDatasetGeneration, mix_dataset, prepare_data_for_metalearning
from utility import (
    get_args, fix_seed, flatten_list,
    numbering_children
)
from generation_model import MyBart
import higher


"""
    This program is a generation task using MyBARTGenerate models(BartForConditionalGeneration)
    Inputs: "S has a <mask> named O."
    Outputs: "S has a child named O."
"""


# main
def main():
    args = get_args()

    device = torch.device(f'cuda:{args.gpu_number}') if torch.cuda.is_available() else torch.device('cpu')
    # 乱数シード値の固定
    fix_seed(42)

    # DataSet
    # numerical settings
    relation_num = args.relation
    children_num = args.children
    batch_size = args.batch_size
    epoch_num = args.epoch_num
    ratio = args.mix_ratio
    lr = args.lr
    numerical_settings = {'relation_num': relation_num,
                          'children_num': children_num,
                          'batch_size': batch_size,
                          'epoch_num': epoch_num,
                          'lr': lr}
    print(numerical_settings)

    files = ['data/1to2Relation_with_id_revised.csv',
             'data/1to3Relation_with_id.csv',
             'data/1to4Relation_with_id.csv']

    org_child_pairs, parent_labels, _org_masked_sentences, _, _ = mix_dataset(files, files, numerical_settings,
                                                                              sentence='masked', one_token=True)

    # oreore tokenizerの都合上、カンマやアポストロフィなどを取り除く
    child_labels = list(flatten_list([eval(child_pair) for child_pair in org_child_pairs]))
    child_labels = [child for child in child_labels]
    child_pairs = [eval(child_pair) for child_pair in org_child_pairs]
    org_child_pairs = [' '.join(eval(child_pair)) for child_pair in org_child_pairs]
    org_masked_sentences = [f'The children of {parent} are <mask>' for parent in parent_labels]

    child_pairs_label = [' '.join(child_pair) for child_pair in child_pairs]
    child_pairs_label = [child_pair for child_pair in child_pairs_label]
    sample_q = org_masked_sentences[50]
    sample_a = child_pairs_label[50]

    print('Dataset Prepared')
    print(f'#Fathers = {len(parent_labels)}, #Children = {len(child_labels)}, #Child_pairs = {len(child_pairs_label)}')

    one2two = 0
    one2three = 0
    one2four = 0
    for child_pair in child_pairs:
        if len(child_pair) == 2:
            one2two += 1
        elif len(child_pair) == 3:
            one2three += 1
        elif len(child_pair) == 4:
            one2four += 1
    print(f'1toNRelations: #1to2 = {one2two}, #1to3 = {one2three}, #1to4 = {one2four}')

    # Training Setup
    print(f'Device: {device}')
    print(sample_q, sample_a)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = MyBart()
    model.to(device)

    # add few shot from 1toN Relations
    thresh = children_num // 3
    show_instances = int((ratio / 100) * thresh)
    print(f'#few shot instances for each: {show_instances}')
    one_two_sent, one_two_label = org_masked_sentences[:show_instances], child_pairs_label[:show_instances]
    one_three_sent = org_masked_sentences[one2two:one2two+show_instances]
    one_three_label = child_pairs_label[one2two:one2two+show_instances]
    one_four_sent = org_masked_sentences[one2two+one2three:one2two+one2three+show_instances]
    one_four_label = child_pairs_label[one2two+one2three:one2two+one2three+show_instances]
    print(f'#fewshot 1toN : #1to2 = {len(one_two_sent)}, #1to3 = {len(one_three_sent)}, #1to4 = {len(one_four_sent)}')
    print(f'ex. 1to2: {one_two_sent[-1]}, {one_two_label[-1]}, {len(one_two_label)}')
    print(f'ex. 1to3: {one_three_sent[0]}, {one_three_label[0]}, {len(one_three_label)}')
    print(f'ex. 1to4: {one_four_sent[0]}, {one_four_label[0]}, {len(one_four_label)}')

    # decompose 1toN relation to 1to1 relations (numbering)
    one2one_sentences, one2one_label = numbering_children(parent_labels, child_pairs, decompose=True)

    print(f'1 to 1 Ex: {one2one_sentences[:2]}, {one2one_label[:2]}, sum={len(one2one_sentences)}')
    print(f'Few Shot Ex: {one_three_sent[0]}, {one_three_label[0]}')

    mix_label = one2one_label + one_two_label + one_three_label + one_four_label
    mix_sentences = one2one_sentences + one_two_sent + one_three_sent + one_four_sent

    mix_label_encoded = tokenizer.batch_encode_plus(mix_label, truncation=True, padding=True, max_length=30,
                                                    add_special_tokens=True, return_tensors='pt')
    mix_sentences_encoded = tokenizer.batch_encode_plus(mix_sentences, truncation=True, padding=True,
                                                        add_special_tokens=True, max_length=30,
                                                        return_tensors='pt')

    mix_train_dataset = MyDatasetGeneration(mix_sentences_encoded, mix_label_encoded)
    mix_train_loader = DataLoader(mix_train_dataset, batch_size=batch_size, shuffle=True)

    meta_train, meta_dev, meta_test = prepare_data_for_metalearning(files, numerical_settings, ratio, tokenizer)
    meta_train_loader = DataLoader(meta_train, batch_size=batch_size, shuffle=True)   # 1to1 + n/2 % few shots
    meta_dev_batch_size = len(meta_dev) // 3
    if meta_dev_batch_size == 0:
        meta_dev_batch_size = 2
    meta_dev_loader = DataLoader(meta_dev, batch_size=meta_dev_batch_size, shuffle=True)   # n/2 % few shots
    meta_test_loader = DataLoader(meta_test, batch_size=batch_size, shuffle=False)   # meta_train, devで見ていない1toN

    print(f'1 to N Ex: {org_masked_sentences[0]}, {child_pairs_label[0]}')
    one2n_labels = tokenizer.batch_encode_plus(child_pairs_label, truncation=True, padding=True, add_special_tokens=True,
                                               max_length=16, return_tensors='pt')
    one2n_encoded = tokenizer.batch_encode_plus(org_masked_sentences, truncation=True, padding=True,
                                                add_special_tokens=True, max_length=32, return_tensors='pt')

    one2n_dataset = MyDatasetGeneration(one2n_encoded, one2n_labels)
    one2n_data_loader = DataLoader(one2n_dataset, batch_size=batch_size, shuffle=True)

    # --------------- dataset for each relations -----------------------
    # 1to2
    one2two_sent, one2two_label = org_masked_sentences[:one2two], child_pairs_label[:one2two]
    print(f'1to2 DATASET: {len(one2two_sent)}')
    one2two_labels = tokenizer.batch_encode_plus(one2two_label, truncation=True, padding=True,
                                                 add_special_tokens=True,
                                                 max_length=16, return_tensors='pt')
    one2two_encoded = tokenizer.batch_encode_plus(one2two_sent, truncation=True, padding=True,
                                                  add_special_tokens=True, max_length=32, return_tensors='pt')

    one2two_dataset = MyDatasetGeneration(one2two_encoded, one2two_labels)
    one2two_data_loader = DataLoader(one2two_dataset, batch_size=batch_size, shuffle=True)

    # 1to3
    one2three_sent, one2three_label = org_masked_sentences[one2two:one2two+one2three], child_pairs_label[one2two:one2two+one2three]
    print(f'1to3 DATASET: {len(one2three_sent)}')
    one2three_labels = tokenizer.batch_encode_plus(one2three_label, truncation=True, padding=True,
                                                   add_special_tokens=True,
                                                   max_length=16, return_tensors='pt')
    one2three_encoded = tokenizer.batch_encode_plus(one2three_sent, truncation=True, padding=True,
                                                    add_special_tokens=True, max_length=32, return_tensors='pt')

    one2three_dataset = MyDatasetGeneration(one2three_encoded, one2three_labels)
    one2three_data_loader = DataLoader(one2three_dataset, batch_size=batch_size, shuffle=True)

    # 1to4
    one2four_sent, one2four_label = org_masked_sentences[-one2four:], child_pairs_label[-one2four:]
    print(f'1to4 DATASET: {len(one2four_sent)}')
    one2four_labels = tokenizer.batch_encode_plus(one2four_label, truncation=True, padding=True,
                                                  add_special_tokens=True,
                                                  max_length=16, return_tensors='pt')
    one2four_encoded = tokenizer.batch_encode_plus(one2four_sent, truncation=True, padding=True,
                                                   add_special_tokens=True, max_length=32, return_tensors='pt')

    one2four_dataset = MyDatasetGeneration(one2four_encoded, one2four_labels)
    one2four_data_loader = DataLoader(one2four_dataset, batch_size=batch_size, shuffle=True)

    print(f'Task1: {org_masked_sentences[0], child_pairs_label[0]}')

    inner_optimizer = AdamW(model.parameters(), lr=lr)
    outer_optimizer = AdamW(model.parameters(), lr=1e-6)

    loss_list = []
    acc_list_one2one = []
    acc_list_1toN = []
    acc_list_1to2 = []
    acc_list_1to3 = []
    acc_list_1to4 = []
    for epoch in tqdm.tqdm(range(epoch_num // len(meta_dev_loader))):
        model.train()
        sum_inner_loss = .0
        sum_outer_loss = .0
        for outer_batch in meta_dev_loader:
            outer_optimizer.zero_grad()

            for inner_batch in meta_train_loader:
                inner_optimizer.zero_grad()
                inner_loss = model(inner_batch, device=device)
                inner_loss.backward()
                inner_optimizer.step()
                sum_inner_loss += inner_loss.item()

            outer_loss = model(outer_batch, device=device)
            outer_loss.backward()
            print('f')
            outer_optimizer.step()
            sum_outer_loss += outer_loss.item()

        final_loss = sum_inner_loss / len(meta_train_loader) + sum_outer_loss / len(meta_dev_loader)

        loss_list.append(final_loss)
        acc_list_one2one.append(model.get_acc(meta_train_loader, device))
        acc_list_1toN.append(model.get_acc(one2n_data_loader, device))
        acc_list_1to2.append(model.get_acc(one2two_data_loader, device))
        acc_list_1to3.append(model.get_acc(one2three_data_loader, device))
        acc_list_1to4.append(model.get_acc(one2four_data_loader, device))

    # Analysis
    model.eval()

    print(f'Acc 1: {max(acc_list_one2one)}')
    print('======== all one2n ========')
    print(model.get_acc(one2n_data_loader, device, verbose=True))
    print('======== one2two =========')
    print(model.get_acc(one2two_data_loader, device, verbose=True))
    print('\n======== one2three =========')
    print(model.get_acc(one2three_data_loader, device, verbose=True))
    print('\n======== one2four =========')
    print(model.get_acc(one2four_data_loader, device, verbose=True))

    accs = [acc_list_one2one, acc_list_1toN, acc_list_1to2, acc_list_1to3, acc_list_1to4]

    labels = ['seen relations(1to1 + few 1toN)', '1toN Relations', '1to2', '1to3', '1to4']
    show_acc_loss(numerical_settings, accs, loss_list,
                  f'../1toN_result/meta_BART_id_numbering_{ratio}%shot_result_{children_num}.png',
                  labels=labels)


if __name__ == '__main__':
    main()
