import torch
import tqdm
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import BartTokenizer

from analysis import show_acc_loss
from data import MyDatasetGeneration, mix_dataset
from utility import (
    get_args, flatten_list,
    numbering_children
)
from constrained_model import BARTDecoderConstrained
from tokenizer import TokenizerForDecoderConstrained


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
    seed = 42
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

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
                                                                              sentence='masked', one_token=False)

    # oreore tokenizerの都合上、カンマやアポストロフィなどを取り除く
    child_labels = list(flatten_list(eval(child_pair) for child_pair in org_child_pairs))
    child_labels = [child.replace("'", '').replace('︱', '') for child in child_labels]
    #org_child_pairs = [string.replace("'", '').replace('︱', '') for string in org_child_pairs]
    child_pairs = [eval(child_pair) for child_pair in org_child_pairs]
    org_child_pairs = [' '.join(eval(child_pair)) for child_pair in org_child_pairs]
    org_masked_sentences = [f'The children of {parent} are <mask>' for parent in parent_labels]
    print(child_pairs[:6])

    # masked_sentences, child_pairs = get_permutations(org_masked_sentences, child_pairs, first=None)

    print('Dataset Prepared')
    print(f'#Fathers = {len(parent_labels)}, #Children = {len(child_labels)}, #Child_pairs = {len(org_child_pairs)}')

    # Training Setup
    print(f'Device: {device}')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    # Prepare decoder tokenizer(oreore tokenizer)
    tokens = ['<s>', '<pad>', '</s>', '<unk>', '<mask>']
    names = child_labels

    tokens.extend(list(set(names)))
    print('len(names): ', len(names), 'len(set(names)): ', len(set(names)))
    tokens2ids = dict(zip(tokens, list(range(len(tokens)))))
    decoder_tokenizer = TokenizerForDecoderConstrained(tokens2ids)
    print(f'Decoder tokenizer prepared.(len={decoder_tokenizer.vocab_size()-5}+5={decoder_tokenizer.vocab_size()})')
    model = BARTDecoderConstrained(decoder_tokenizer, decoder_tokenizer.vocab_size())
    model.to(device)

    # decompose 1toN relation to 1 to 1 relations
    one2n_masked_sentences, one2n_label = numbering_children(parent_labels, child_pairs, decompose=False)
    print(f'Aggregated Ex: {one2n_masked_sentences[:3]}, {one2n_label[:3]}')

    sample_q = one2n_masked_sentences[50]
    sample_a = one2n_label[50]

    # add few shot from 1toN Relations
    thresh = children_num // 3
    mix_ratio = (ratio // 10) * (thresh // 10)
    one_two_sent, one_two_label = one2n_masked_sentences[:mix_ratio], one2n_label[:mix_ratio]
    one_three_sent = one2n_masked_sentences[thresh:thresh+mix_ratio]
    one_three_label = one2n_label[thresh:thresh+mix_ratio]
    one_four_sent = one2n_masked_sentences[thresh*2:thresh*2+mix_ratio]
    one_four_label = one2n_label[thresh*2:thresh*2+mix_ratio]
    print(f'fewshot_1toN: #1to2 = {len(one_two_sent)}, #1to3 = {len(one_three_sent)}, #1to4 = {len(one_four_sent)}')
    print(f'ex. 1to2: {one_two_sent[-1]}, {one_two_label[-1]}, {len(one_two_label)}')
    print(f'ex. 1to3: {one_three_sent[0]}, {one_three_label[0]}, {len(one_three_label)}')
    print(f'ex. 1to4: {one_four_sent[0]}, {one_four_label[0]}, {len(one_four_label)}')

    child_labels_with_headspace = [' '+child for child in child_labels]
    print(f'child_labels: {child_labels[:3]}')
    print(f'child_labels_with_space: {child_labels_with_headspace[:3]}')

    seen_label = one_two_label + one_three_label + one_four_label
    seen_masked_sentences = one_two_sent + one_three_sent + one_four_sent

    print(f'Show sentence: {seen_masked_sentences[0]}')
    print(f'Show label: {seen_label[0]}')

    seen_label_encoded = decoder_tokenizer.tokenize(seen_label, pad_length=15)
    seen_sentences_encoded = tokenizer.batch_encode_plus(seen_masked_sentences, truncation=True, padding=True,
                                                         add_special_tokens=True, max_length=60,
                                                         return_tensors='pt')

    seen_train_dataset = MyDatasetGeneration(seen_sentences_encoded,
                                             seen_label_encoded)
    seen_train_loader = DataLoader(seen_train_dataset, batch_size=batch_size, shuffle=True)

    # prepare valuation loader for task1(permutations)
    eval_labels = decoder_tokenizer.tokenize(one2n_label, pad_length=20)
    eval_encoded = tokenizer.batch_encode_plus(one2n_masked_sentences,
                                               truncation=True, padding=True,
                                               add_special_tokens=True, max_length=60, return_tensors='pt')
    eval_dataset = MyDatasetGeneration(eval_encoded, eval_labels)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr)

    # Prepare for name memorization
    name_label_encoded = decoder_tokenizer.tokenize(child_labels+child_labels, pad_length=2)
    name_input_encoded = tokenizer.batch_encode_plus(child_labels+child_labels_with_headspace, truncation=True,
                                                     padding=True, add_special_tokens=True, max_length=7,
                                                     return_tensors='pt')

    name_train_loader = DataLoader(MyDatasetGeneration(name_input_encoded, name_label_encoded),
                                   batch_size=batch_size, shuffle=True)

    _loss_name_mem = []
    _acc_name_mem = []
    for epoch in tqdm.tqdm(range(150)):
        model.train()
        running_loss = .0
        for batch in name_train_loader:
            optimizer.zero_grad()
            loss = model(batch, device)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        _loss_name_mem.append(running_loss / len(batch))
        model.eval()
        _acc_name_mem.append(model.get_acc(name_train_loader, device))



    loss_list = []
    acc_list_seen = []
    acc_list_all = []

    for epoch in tqdm.tqdm(range(epoch_num)):
        model.train()
        running_loss = .0
        for batch in seen_train_loader:
            optimizer.zero_grad()
            loss = model(batch, device)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        loss_list.append(running_loss / len(batch))
        model.eval()
        acc_list_seen.append(model.get_acc(seen_train_loader, device))
        acc_list_all.append(model.get_acc(eval_loader, device))
    print(f'Name memorization is done. Acc. :{_acc_name_mem[-1]}')

    # Analysis
    model.eval()

    print(f'Acc 1: {max(acc_list_seen)}')
    # models.get_acc(train_loader1, device, verbose=True)

    show_acc_loss(numerical_settings, [acc_list_seen, acc_list_all], loss_list,
                  f'../1toN_result/ConstrainedBART_id_StepbyStep_NameMemorize_copying_{ratio}%shot_result_{children_num}.png')

    tok = tokenizer(sample_q, return_tensors='pt')
    model.plot_prob_scores(tok['input_ids'], sample_a,
                           f'../1toN_result/ConstrainedBART_id_StepbyStep_NameMemorize_copying_seen_{ratio}%shot_{children_num}_Scores.png', device)


if __name__ == '__main__':
    main()
