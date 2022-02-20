import torch
import re
from tqdm import tqdm
from torch.optim import SGD
from torch.utils.data import DataLoader

from rnn_data import mix_dataset, prepare_data_for_metalearning
from rnn_tokenizer import MyTokenizer
from rnn_model import EncoderRNN, AttnDecoderRNN
from rnn_train import train
from rnn_analysis import evaluate

from rnn_utility import get_args, fix_seed, flatten_list


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

    src_txt = ['<s>', '</s>', '<pad>', '<unk>', 'Who', 'is', 'are', 'the',
               'first', 'second', 'third', 'fourth', 'child', 'children', 'of', '?']
    # parentの名前から 「,」と「.」を除外
    src_txt.extend(list(set(list(flatten_list(
        [parent.replace(',', '').replace('.', '').replace("'", '').split(' ') for parent in parent_labels])))))
    tgt_txt = ['<s>', '</s>', '<pad>', '<unk>']
    org_child_pairs = list(flatten_list([eval(child_pair) for child_pair in org_child_pairs]))
    tgt_txt.extend(org_child_pairs)

    # src_input_lines, tgt_input_lines = read_vocab(src_txt, tgt_txt)
    # src_word2index, tgt_word2index = src_input_lines.word2index, tgt_input_lines.word2index
    src_word2index = dict(zip(src_txt, list(range(len(src_txt)))))
    tgt_word2index = dict(zip(tgt_txt, list(range(len(tgt_txt)))))
    src_index2word = dict(zip(src_word2index.values(), src_word2index.keys()))
    tgt_index2word = dict(zip(tgt_word2index.values(), tgt_word2index.keys()))
    src_tokenizer, tgt_tokenizer = MyTokenizer(src_word2index), MyTokenizer(tgt_word2index)

    train_dataset, meta_dataset, test_dataset = prepare_data_for_metalearning(files, numerical_settings, ratio,
                                                                              src_tokenizer, tgt_tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    meta_loader = DataLoader(meta_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    hidden_size = 256
    encoder = EncoderRNN(device, len(src_word2index)+5, hidden_size).to(device)
    attn_decoder = AttnDecoderRNN(device, hidden_size, len(tgt_word2index)+5, dropout_p=0.1).to(device)

    print(f'Encoder Vocab. size: {len(src_word2index)}')
    print(f'Decoder Vocab. size: {len(tgt_word2index)}')

    learning_rate = 1e-5
    encoder_optimizer = SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = SGD(attn_decoder.parameters(), lr=learning_rate)
    criterion = torch.nn.NLLLoss()

    loss = []
    train_acc = []
    print(src_tokenizer.tokenize([f'Who is the first child of {parent_labels[0]}?']))
    print('学習かいし')
    for _epoch in tqdm(range(args.epoch_num)):
        running_loss = .0
        for batch in train_loader:
            input_ids = batch['input_ids'].view(-1, 1)
            label = batch['labels'].view(-1, 1)

            running_loss += train(device, input_ids, label, encoder, attn_decoder,
                                  encoder_optimizer, decoder_optimizer, criterion)
        running_loss / len(batch)
        loss.append(running_loss)

        # evaluate (get accuracy)
        acc = .0
        for batch in train_loader:
            input_ids = batch['input_ids'].view(-1, 1)
            label = batch['labels'].view(-1)
            decoded_tokens = evaluate(device, encoder, attn_decoder, input_ids)
            print(type(decoded_tokens), decoded_tokens)
            print(type(label.tolist()), label.tolist())
            est = set(decoded_tokens) - {1, 2, 3, 4}
            gold = set(label.tolist()) - {1, 2, 3, 4}
            if est == gold:
                acc += 1
        acc = acc / len(train_loader)

        train_acc.append(acc)
    print(loss)
    print(train_acc)


if __name__ == '__main__':
    main()
