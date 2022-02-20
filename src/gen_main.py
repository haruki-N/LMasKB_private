import torch
import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AdamW
from transformers import BartTokenizer

from analysis import show_acc_loss
from data import (
    MyDatasetGeneration,
    mix_dataset,
    prepare_dataloader,
    prepare_dataset,
    split_train_valid_test
)
from utility import (
    get_args, fix_seed, flatten_list,
    numbering_children
)
from generation_model import MyBart


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
    relation_num = 0
    children_num = args.father
    batch_size = args.batch_size
    epoch_num = args.epoch_num
    ratio = args.mix_ratio
    lr = args.lr
    freeze_encoder = args.freeze_encoder
    second_task = args.second_task
    numerical_settings = {'relation_num': relation_num,
                          'father_num': children_num,
                          'batch_size': batch_size,
                          'epoch_num': epoch_num,
                          'lr': lr,
                          'freeze_encoder': freeze_encoder,
                          'second_task': second_task}
    print(numerical_settings)

    files = ['data/1to2Relation_with_id_revised.csv',
             'data/1to3Relation_with_id.csv',
             'data/1to4Relation_with_id.csv']

    org_child_pairs, parent_labels, _org_masked_sentences = mix_dataset(files, numerical_settings, one_token=True)

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

    # split data into train, valid, test
    """
    train_ratio = 30
    valid_ratio = 30
    test_ratio = 40
    train, valid, test = split_train_valid_test(org_masked_sentences, child_pairs, train_ratio, valid_ratio, test_ratio)
    train_parent, train_child = train
    valid_parent, valid_child = valid
    test_parent, test_child = test
    train_child_label = [' '.join(child) for child in train_child]
    valid_child_label = [' '.join(child) for child in valid_child]
    test_child_label = [' '.join(child) for child in test_child]
    """

    # decompose 1toN relation to 1to1 relations (numbering) & prepare dataloader for 1to1 relations
    one2one_sentences, one2one_label = numbering_children(parent_labels, child_pairs, decompose=True)
    one2one_data_loader = prepare_dataloader(one2one_sentences, one2one_label, input_pad_length=24, label_pad_length=16,
                                             batch_size=batch_size, tokenizer=tokenizer, shuffle=True)

    # prepare train, valid, test dataloader
    train_inputs = one2one_sentences # + train_parent ← for train,dev,test split
    train_label = one2one_label # + train_child_label ← for train,dev,test split
    train_loader = prepare_dataloader(train_inputs, train_label, input_pad_length=24, label_pad_length=16,
                                      batch_size=batch_size, tokenizer=tokenizer, shuffle=True)

    # valid 1to1 mixing
    """
    valid_one2one_sent, valid_one2one_label = numbering_children(valid_parent, valid_child, decompose=True)
    valid_mixing_parent = valid_one2one_sent + valid_parent
    valid_mixing_label = valid_one2one_label + valid_child_label
    valid_loader = prepare_dataloader(valid_parent, valid_child_label, input_pad_length=24, label_pad_length=16,
                                      tokenizer=tokenizer, batch_size=batch_size, shuffle=True)
    test_loader = prepare_dataloader(test_parent, test_child_label, input_pad_length=24, label_pad_length=16,
                                     tokenizer=tokenizer, batch_size=batch_size, shuffle=True)
    """

    # prepare 1toN few shot dataloader
    fewshots_labels = one_two_label + one_three_label + one_four_label
    fewshots_sentences = one_two_sent + one_three_sent + one_four_sent
    fewshots_batch = len(fewshots_sentences) // 3
    if fewshots_batch == 0:
        fewshots_batch = 8
    fewshots_data_loader = prepare_dataloader(fewshots_sentences, fewshots_labels, input_pad_length=24, shuffle=True,
                                              label_pad_length=16, batch_size=fewshots_batch, tokenizer=tokenizer)

    print(f'1 to 1 Ex: {one2one_sentences[:2]}, {one2one_label[:2]}, sum={len(one2one_sentences)}')
    print(f'Few Shot Ex: {one_three_sent[0]}, {one_three_label[0]}')

    mix_label = one2one_label + one_two_label + one_three_label + one_four_label
    mix_sentences = one2one_sentences + one_two_sent + one_three_sent + one_four_sent
    mix_train_loader = prepare_dataloader(mix_sentences, mix_label, input_pad_length=24, label_pad_length=16,
                                          batch_size=batch_size, tokenizer=tokenizer, shuffle=True)

    print(f'1 to N Ex: {org_masked_sentences[0]}, {child_pairs_label[0]}')
    one2n_data_loader = prepare_dataloader(org_masked_sentences, child_pairs_label, input_pad_length=24,
                                           label_pad_length=16, batch_size=batch_size, tokenizer=tokenizer, shuffle=True)

    # --------------- dataset for each relations -----------------------
    # 1to2
    one2two_sent, one2two_label = org_masked_sentences[:one2two], child_pairs_label[:one2two]
    print(f'1to2 DATASET: {len(one2two_sent)}')
    one2two_data_loader = prepare_dataloader(one2two_sent, one2two_label, input_pad_length=24, label_pad_length=16,
                                             batch_size=batch_size, tokenizer=tokenizer, shuffle=True)

    # 1to3
    one2three_sent, one2three_label = org_masked_sentences[one2two:one2two+one2three], child_pairs_label[one2two:one2two+one2three]
    print(f'1to3 DATASET: {len(one2three_sent)}')
    one2three_data_loader = prepare_dataloader(one2three_sent, one2three_label, input_pad_length=24, label_pad_length=16,
                                               batch_size=batch_size, tokenizer=tokenizer, shuffle=True)

    # 1to4
    one2four_sent, one2four_label = org_masked_sentences[-one2four:], child_pairs_label[-one2four:]
    print(f'1to4 DATASET: {len(one2four_sent)}')
    one2four_data_loader = prepare_dataloader(one2four_sent, one2four_label, input_pad_length=24, label_pad_length=16,
                                              batch_size=batch_size, tokenizer=tokenizer, shuffle=True)

    print(f'Task1: {org_masked_sentences[0], child_pairs_label[0]}')

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)

    train_loss = []
    # valid_loss_list = []
    acc_list_one2one = []
    acc_list_1toN = []
    acc_list_1to2 = []
    acc_list_1to3 = []
    acc_list_1to4 = []
    # train_acc = []
    # valid_acc = []
    # test_acc = []
    for _epoch in tqdm.tqdm(range(epoch_num)):
        model.train()
        running_loss = .0
        # running_valid_loss = .0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(batch, device)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        """
        valid_loss = .0
        for valid in valid_loader:
            valid_loss += model(valid, device).detach().item()
        valid_loss /= len(valid_loader)
        running_valid_loss += valid_loss
        scheduler.step(valid_loss)
        """

        train_loss.append(running_loss / len(train_loader))
        # valid_loss_list.append(running_valid_loss / len(train_loader))
        model.eval()
        # train_acc.append(model.get_acc(train_loader, device))
        # valid_acc.append(model.get_acc(valid_loader, device))
        # test_acc.append(model.get_acc(test_loader, device))
        acc_list_one2one.append(model.get_acc(mix_train_loader, device))
        acc_list_1toN.append(model.get_acc(one2n_data_loader, device))
        acc_list_1to2.append(model.get_acc(one2two_data_loader, device))
        acc_list_1to3.append(model.get_acc(one2three_data_loader, device))
        acc_list_1to4.append(model.get_acc(one2four_data_loader, device))

    """
    if second_task:
        # --------- freeze encoder params ---------
        if freeze_encoder:
            for name, param in model.named_parameters():
                if 'decoder' not in name:
                    param.requires_grad = False

            print('check if the encoder params are freezed')
            freezed_param = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    freezed_param.append(name)
            print(f'FREEZED: {freezed_param}')
        # -----------------------------------------
        if second_task == 'mixing':
            second_loader = mix_train_loader
        elif second_task == 'fewshots':
            second_loader = fewshots_data_loader
        for _epoch in tqdm.tqdm(range(epoch_num)):
            model.train()
            running_loss = .0
            for batch in second_loader:
                optimizer.zero_grad()
                loss = model(batch, device)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()

            loss_list.append(running_loss / len(batch))
            model.eval()
            acc_list_one2one.append(model.get_acc(mix_train_loader, device))
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
    """

    accs = [acc_list_one2one, acc_list_1toN, acc_list_1to2, acc_list_1to3, acc_list_1to4]
    # accs = [train_acc, valid_acc, test_acc]

    labels = ['seen relations(1to1 + few 1toN)', '1toN Relations', '1to2', '1to3', '1to4']
    # labels = ['train(all 1to1 + 30%1toN)', 'valid(30%1toN)', 'test(40%1toN)']

    if second_task:
        title = f'../1toN_result/BART_id_one2one_then_{second_task}_numbering_{ratio}%shot_result_{children_num*3}.png'
        if freeze_encoder:
            title = f'../1toN_result/BART_id_one2one_then_EncFreezed_{second_task}_numbering_{ratio}%shot_result_{children_num*3}.png'
    else:
        title = f'../1toN_result/BART_id_numbering_result_default_{children_num*3}_{batch_size}_{lr}.png'
    print(title)
    show_acc_loss(numerical_settings, accs, train_loss, title, labels=labels)


if __name__ == '__main__':
    main()
