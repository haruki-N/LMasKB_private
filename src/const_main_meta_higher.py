import torch
import tqdm
import sys
from torch.utils.data import DataLoader
from transformers import BartTokenizer

from analysis import show_acc_loss
from data import MyDatasetGeneration, mix_dataset, prepare_data_for_metalearning
from utility import (
    get_args, fix_seed, flatten_list,
    numbering_children
)
from constrained_model import BARTDecoderConstrained
from tokenizer import TokenizerForDecoderConstrained
import higher


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
    freeze_encoder = args.freeze_encoder
    numerical_settings = {'relation_num': relation_num,
                          'children_num': children_num,
                          'batch_size': batch_size,
                          'epoch_num': epoch_num,
                          'lr': lr,
                          'freeze_encoder': freeze_encoder}
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
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    # Prepare decoder tokenizer(oreore tokenizer)
    tokens = ['<s>', '<pad>', '</s>', '<unk>', '<mask>']
    names = child_labels

    tokens.extend(list(set(names)))
    print('len(child_names): ', len(names), 'len(set(child_names)): ', len(set(names)))
    tokens2ids = dict(zip(tokens, list(range(len(tokens)))))
    decoder_tokenizer = TokenizerForDecoderConstrained(tokens2ids)
    print(f'Decoder tokenizer prepared.(len={decoder_tokenizer.vocab_size() - 5}+5={decoder_tokenizer.vocab_size()})')
    model = BARTDecoderConstrained(decoder_tokenizer, decoder_tokenizer.vocab_size())
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
    one2one_label_encoded = decoder_tokenizer.tokenize(one2one_label, pad_length=4)
    one2one_sentence_encoded = tokenizer.batch_encode_plus(one2one_sentences, truncation=True, add_special_tokens=True,
                                                           padding=True, max_length=30, return_tensors='pt')
    one2one_dataset = MyDatasetGeneration(one2one_sentence_encoded, one2one_label_encoded)
    one2one_data_loader = DataLoader(one2one_dataset, batch_size=batch_size, shuffle=True)

    print(f'1 to 1 Ex: {one2one_sentences[:2]}, {one2one_label[:2]}, sum={len(one2one_sentences)}')
    print(f'Few Shot Ex: {one_three_sent[0]}, {one_three_label[0]}')

    mix_label = one2one_label + one_two_label + one_three_label + one_four_label
    mix_sentences = one2one_sentences + one_two_sent + one_three_sent + one_four_sent

    mix_label_encoded = decoder_tokenizer.tokenize(mix_label, pad_length=8)
    mix_sentences_encoded = tokenizer.batch_encode_plus(mix_sentences, truncation=True, padding=True,
                                                        add_special_tokens=True, max_length=30,
                                                        return_tensors='pt')

    mix_train_dataset = MyDatasetGeneration(mix_sentences_encoded, mix_label_encoded)
    mix_train_loader = DataLoader(mix_train_dataset, batch_size=batch_size, shuffle=True)

    meta_train_loader, meta_dev_loader, meta_test_loader = prepare_data_for_metalearning(files, numerical_settings,
                                                                                         ratio, batch_size,
                                                                                         tokenizer, decoder_tokenizer)

    print(f'1 to N Ex: {org_masked_sentences[0]}, {child_pairs_label[0]}')
    one2n_labels = decoder_tokenizer.tokenize(child_pairs_label, pad_length=8)
    one2n_encoded = tokenizer.batch_encode_plus(org_masked_sentences, truncation=True, padding=True,
                                                add_special_tokens=True, max_length=32, return_tensors='pt')

    one2n_dataset = MyDatasetGeneration(one2n_encoded, one2n_labels)
    one2n_data_loader = DataLoader(one2n_dataset, batch_size=batch_size, shuffle=True)

    # --------------- dataset for each relations -----------------------
    # 1to2
    one2two_sent, one2two_label = org_masked_sentences[:one2two], child_pairs_label[:one2two]
    print(f'1to2 DATASET: {len(one2two_sent)}')
    one2two_labels = decoder_tokenizer.tokenize(one2two_label, pad_length=8)
    one2two_encoded = tokenizer.batch_encode_plus(one2two_sent, truncation=True, padding=True,
                                                  add_special_tokens=True, max_length=32, return_tensors='pt')

    one2two_dataset = MyDatasetGeneration(one2two_encoded, one2two_labels)
    one2two_data_loader = DataLoader(one2two_dataset, batch_size=batch_size, shuffle=True)

    # 1to3
    one2three_sent, one2three_label = org_masked_sentences[one2two:one2two+one2three], child_pairs_label[one2two:one2two+one2three]
    print(f'1to3 DATASET: {len(one2three_sent)}')
    one2three_labels = decoder_tokenizer.tokenize(one2three_label, pad_length=8)
    one2three_encoded = tokenizer.batch_encode_plus(one2three_sent, truncation=True, padding=True,
                                                    add_special_tokens=True, max_length=32, return_tensors='pt')

    one2three_dataset = MyDatasetGeneration(one2three_encoded, one2three_labels)
    one2three_data_loader = DataLoader(one2three_dataset, batch_size=batch_size, shuffle=True)

    # 1to4
    one2four_sent, one2four_label = org_masked_sentences[-one2four:], child_pairs_label[-one2four:]
    print(f'1to4 DATASET: {len(one2four_sent)}')
    one2four_labels = decoder_tokenizer.tokenize(one2four_label, pad_length=8)
    one2four_encoded = tokenizer.batch_encode_plus(one2four_sent, truncation=True, padding=True,
                                                   add_special_tokens=True, max_length=32, return_tensors='pt')

    one2four_dataset = MyDatasetGeneration(one2four_encoded, one2four_labels)
    one2four_data_loader = DataLoader(one2four_dataset, batch_size=batch_size, shuffle=True)

    print(f'Task1: {org_masked_sentences[0], child_pairs_label[0]}')

    inner_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    outer_optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    loss_list = []
    acc_list_1to1 = []
    acc_list_1toN = []
    acc_list_1to2 = []
    acc_list_1to3 = []
    acc_list_1to4 = []

    for _epoch in tqdm.tqdm(range(epoch_num)):
        model.train()
        running_loss = .0
        for batch in one2one_data_loader:
            inner_optimizer.zero_grad()
            loss = model(batch, device)
            running_loss += loss.item()
            loss.backward()
            inner_optimizer.step()

        loss_list.append(running_loss / len(batch))
        model.eval()
        acc_list_1to1.append(model.get_acc(one2one_data_loader, device))
        acc_list_1toN.append(model.get_acc(one2n_data_loader, device))
        acc_list_1to2.append(model.get_acc(one2two_data_loader, device))
        acc_list_1to3.append(model.get_acc(one2three_data_loader, device))
        acc_list_1to4.append(model.get_acc(one2four_data_loader, device))

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

    for _epoch in tqdm.tqdm(range(epoch_num)):
        model.train()

        for inner_batch in meta_train_loader:
            with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False, device=device
                 ) as (fmodel, diffopt), torch.backends.cudnn.flags(enabled=False):
                inner_loss = fmodel(inner_batch, device=device)
                diffopt.step(inner_loss)

                mean_outer_loss = torch.Tensor([0.0]).to(device)
                with torch.set_grad_enabled(model.training):
                    for outer_batch in meta_dev_loader:
                        mean_outer_loss += fmodel(outer_batch, device=device)

                    mean_outer_loss.div_(len(meta_dev_loader))
                final_loss = inner_loss + mean_outer_loss
                final_loss.backward()

            outer_optimizer.step()
            outer_optimizer.zero_grad()

        loss_list.append(final_loss.item())   # 注意：final_loss.item()にしないと、計算グラフ保持されたfinal_lossがappendされるので、epochごとにメモリ増える
        model.eval()
        acc_list_1to1.append(model.get_acc(one2one_data_loader, device))
        acc_list_1toN.append(model.get_acc(one2n_data_loader, device))
        acc_list_1to2.append(model.get_acc(one2two_data_loader, device))
        acc_list_1to3.append(model.get_acc(one2three_data_loader, device))
        acc_list_1to4.append(model.get_acc(one2four_data_loader, device))

    # Analysis
    model.eval()

    print(f'Acc 1: {max(acc_list_1to1)}')
    print('======== all one2one ========')
    print(model.get_acc(one2one_data_loader, device, verbose=True))
    print('======== one2two =========')
    print(model.get_acc(one2two_data_loader, device, verbose=True))
    print('\n======== one2three =========')
    print(model.get_acc(one2three_data_loader, device, verbose=True))
    print('\n======== one2four =========')
    print(model.get_acc(one2four_data_loader, device, verbose=True))

    accs = [acc_list_1to1, acc_list_1to2, acc_list_1to3, acc_list_1to4, acc_list_1toN]

    labels = ['1to1', '1to2', '1to3', '1to4', '1toN Relations']
    title = f'../1toN_result/one2one_then_meta_higher_constBART_id_numbering_{ratio}%shot_result_{children_num}.png'
    if freeze_encoder:
        title = f'../1toN_result/one2one_then_EncFreeze_meta_higher_constBART_id_numbering_{ratio}%shot_result_{children_num}.png'
    show_acc_loss(numerical_settings, accs, loss_list,
                  title,
                  labels=labels)


if __name__ == '__main__':
    main()
