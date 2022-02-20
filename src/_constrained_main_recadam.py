import torch
import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AdamW
from transformers import BartTokenizer

from analysis import show_acc_loss
from data import MyDatasetGeneration, mix_dataset
from utility import (
    get_args, fix_seed, flatten_list,
    numbering_children
)
from constrained_model import BARTDecoderConstrained
from tokenizer import TokenizerForDecoderConstrained
from optimizer.RecAdam import RecAdam


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
    child_labels = list(flatten_list(eval(child_pair) for child_pair in org_child_pairs))
    child_labels = [child.replace('"', '').replace("'", '').replace('︱', '') for child in child_labels]
    child_pairs = [eval(child_pair) for child_pair in org_child_pairs]
    org_child_pairs = [' '.join(eval(child_pair)) for child_pair in org_child_pairs]
    org_masked_sentences = [f'The children of {parent} are <mask>' for parent in parent_labels]

    # masked_sentences, child_pairs = get_permutations(org_masked_sentences, child_pairs, first=None)

    child_pairs_label = [' '.join(child_pair) for child_pair in child_pairs]
    child_pairs_label = [child_pair.replace('"', '').replace("'", '').replace('︱', '')
                         for child_pair in child_pairs_label]
    sample_q = org_masked_sentences[50]
    sample_a = child_pairs_label[50]

    print('Dataset Prepared')

    # Training Setup
    print(f'Device: {device}')
    print(sample_q, sample_a)
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

    # add few shot from 1toN Relations
    thresh = children_num // 3
    mix_ratio = (ratio // 10) * (thresh // 10)
    one_two_sent, one_two_label = org_masked_sentences[:mix_ratio], child_pairs_label[:mix_ratio]
    one_three_sent = org_masked_sentences[thresh:thresh+mix_ratio]
    one_three_label = child_pairs_label[thresh:thresh+mix_ratio]
    one_four_sent = org_masked_sentences[thresh*2:thresh*2+mix_ratio]
    one_four_label = child_pairs_label[thresh*2:thresh*2+mix_ratio]

    # decompose 1toN relation to 1 to 1 relations
    one2one_sentences, one2one_label = numbering_children(parent_labels, child_pairs, decompose=True)
    # one2one_sentences, one2one_label = decompose_relation(parent_labels, child_pairs)
    print(f'1to1 Ex: {one2one_sentences[:2]}, {one2one_label[:2]}')

    # one2one_label += one_two_label + one_three_label + one_four_label
    # one2one_sentences += one_two_sent + one_three_sent + one_four_sent

    one2one_label_encoded = decoder_tokenizer.tokenize(one2one_label, pad_length=8)
    one2one_sentences_encoded = tokenizer.batch_encode_plus(one2one_sentences, truncation=True, padding=True,
                                                            add_special_tokens=True, max_length=24, return_tensors='pt')
    one2one_train_dataset = MyDatasetGeneration(one2one_sentences_encoded, one2one_label_encoded)
    one2one_train_loader = DataLoader(one2one_train_dataset, batch_size=batch_size, shuffle=True)

    # few shot loader
    print(f'Few shot Ex: {one_two_sent[0]} → {one_two_label[0]}')
    few_shots_label = one_two_label + one_three_label + one_four_label
    few_shots_sentence = one_two_sent + one_three_sent + one_four_sent
    few_shots_label_encoded = decoder_tokenizer.tokenize(few_shots_label, pad_length=8)
    few_shots_sent_encoded = tokenizer.batch_encode_plus(few_shots_sentence, truncation=True, padding=True,
                                                         add_special_tokens=True, max_length=24, return_tensors='pt')
    fewshot_dataset = MyDatasetGeneration(few_shots_sent_encoded, few_shots_label_encoded)
    fewshot_train_loader = DataLoader(fewshot_dataset, batch_size=1, shuffle=True)

    print(f'1 to N Ex: {org_masked_sentences[0]}, {child_pairs_label[0]}')
    one2n_labels = decoder_tokenizer.tokenize(child_pairs_label, pad_length=8)
    one2n_encoded = tokenizer.batch_encode_plus(org_masked_sentences, truncation=True, padding=True,
                                                add_special_tokens=True, max_length=24, return_tensors='pt')

    one2n_train_dataset = MyDatasetGeneration(one2n_encoded, one2n_labels)
    one2n_train_loader = DataLoader(one2n_train_dataset, batch_size=batch_size, shuffle=True)

    print(f'Task1: {org_masked_sentences[0], child_pairs_label[0]}')

    mix_labels = one2one_label + few_shots_label
    mix_sentences = one2one_sentences + few_shots_sentence
    mix_label_encoded = decoder_tokenizer.tokenize(mix_labels, pad_length=8)
    mix_sentence_encoded = tokenizer.batch_encode_plus(mix_sentences, truncation=True, padding=True,
                                                       add_special_tokens=True, max_length=24, return_tensors='pt')
    mix_train_dataset = MyDatasetGeneration(mix_sentence_encoded, mix_label_encoded)
    mix_train_loader = DataLoader(mix_train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)

    loss_list = []
    acc_one2one = []
    acc_1toN_fewshot = []
    acc_1toN_all = []

    for _epoch in tqdm.tqdm(range(epoch_num)):
        model.train()
        running_loss = .0
        for batch in mix_train_loader:
            optimizer.zero_grad()
            loss = model(batch, device)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        loss_list.append(running_loss / len(batch))
        acc_one2one.append(model.get_acc(one2one_train_loader, device, one2one=True))
        acc_1toN_fewshot.append(model.get_acc(fewshot_train_loader, device))
        acc_1toN_all.append(model.get_acc(one2n_train_loader, device))

    print(model.get_acc(one2one_train_loader, device, verbose=True, one2one=True))
    optimizer = RecAdam(model.parameters(), lr=lr, pretrain_params=model.parameters())

    for _epoch in tqdm.tqdm(range(100)):
        model.train()
        running_loss = .0
        for batch in fewshot_train_loader:
            optimizer.zero_grad()
            loss = model(batch, device)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        loss_list.append(running_loss / len(batch))
        acc_one2one.append(model.get_acc(one2one_train_loader, device))
        acc_1toN_fewshot.append(model.get_acc(fewshot_train_loader, device))
        acc_1toN_all.append(model.get_acc(one2n_train_loader, device))

    # Analysis
    model.eval()
    labels = ['1to1', 'fewshot 1toN(seen)', '1toN(all)']
    acc_list = [acc_one2one, acc_1toN_fewshot, acc_1toN_all]
    title = f'../1toN_result/ConstrainedBART_id_one2one_then_fewshot_RecAdam_numbering_{ratio}%shot_result_{children_num}.png'

    show_acc_loss(numerical_settings, acc_list, loss_list, title, labels=labels)
    # show_acc_loss(numerical_settings, [acc_one2one[-10:], acc_1toN_fewshot[-10:], acc_1toN_all[-10:]], loss_list,
                  # f'../1toN_result/ConstrainedBART_id_one2one_then_mixing_fewshot_numbering_{ratio}%shot_result_{children_num}2.png',
                  # labels=['1to1', 'fewshot 1toN(seen)', '1toN(all)'])


if __name__ == '__main__':
    main()
