import torch
import tqdm
import sys
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from transformers.optimization import AdamW

from analysis import show_acc_loss
from data import mix_dataset, archived_prepare_data_for_metalearning
from utility import (
    get_args, fix_seed, flatten_list
)
from generation_model import MyBart
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
    numerical_settings = {'N': relation_num,
                          'relation_num': children_num,
                          'children_num': children_num,
                          'batch_size': batch_size,
                          'epoch_num': epoch_num,
                          'lr': lr}
    print(numerical_settings)

    files = ['data/1to2Relation_with_id_revised.csv',
             'data/1to3Relation_with_id.csv',
             'data/1to4Relation_with_id.csv']

    # org_child_pairs, parent_labels, _org_masked_sentences, _, _ = mix_dataset(files, files, numerical_settings,
                                                                              # sentence='masked', one_token=True)

    # Training Setup
    print(f'Device: {device}')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    """
    # Prepare decoder tokenizer(oreore tokenizer)
    tokens = ['<s>', '<pad>', '</s>', '<unk>', '<mask>']
    all_child_names = list(flatten_list([eval(child_pair) for child_pair in org_child_pairs]))
    tokens.extend(list(set(all_child_names)))
    print('len(names): ', len(all_child_names), 'len(set(names)): ', len(set(all_child_names)))
    tokens2ids = dict(zip(tokens, list(range(len(tokens)))))
    decoder_tokenizer = TokenizerForDecoderConstrained(tokens2ids)
    print(f'Decoder tokenizer prepared.(len={decoder_tokenizer.vocab_size()-5}+5={decoder_tokenizer.vocab_size()})')
    """

    # model = BARTDecoderConstrained(decoder_tokenizer, decoder_tokenizer.vocab_size())
    model = MyBart()
    model.to(device)
    # x = input('モデルがGPUにロードされました。続けますかy/n')
    # if x == 'n':
        # sys.exit()

    train_dataset, meta_dataset, test_dataset = archived_prepare_data_for_metalearning(files, numerical_settings, ratio, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    meta_loader = DataLoader(meta_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    outer_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    inner_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    inner_optimizer = AdamW(model.parameters(), lr=lr)

    # outer_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # inner_optimizer = torch.optim.SGD(model.parameters(), lr=lr)


    loss_list = []
    train_acc = []
    meta_acc = []
    test_acc = []
    print(f'Epoch: {epoch_num}, outer_loop: {len(meta_loader)}, inner_loop: {len(train_loader)}')

    for _epoch in tqdm.tqdm(range(epoch_num)):
        model.train()
        running_loss = .0
        for inner_batch in train_loader:
            final_loss = model(inner_batch, device)
            running_loss += final_loss.item()
            final_loss.backward()
            inner_optimizer.step()

        loss_list.append(final_loss.item())
        model.eval()
        train_acc.append(model.get_acc(train_loader, device))
        meta_acc.append(model.get_acc(meta_loader, device))
        test_acc.append(model.get_acc(test_loader, device))

        """
        for inner_batch in train_loader:
            with higher.innerloop_ctx(
                    model, inner_optimizer, copy_initial_weights=False, device=device
            ) as (fmodel, diffopt), torch.backends.cudnn.flags(enabled=False):
                inner_loss = fmodel(inner_batch, device=device)   # no need to call loss.backward()
                # ここにbreakpointを用いて、メモリを逐一確認すべし → nvidia-smi = step-through
                # profilerっていう本格的なやつもある.
                # pytorch的に環境変数宣言でもメモリ確認可能
                diffopt.step(inner_loss)

                # x = input('innerbatchが1周しました。続けますかy/n')
                # if x == 'n':
                    # sys.exit()

                # 
                mean_outer_loss = torch.Tensor([0.0]).to(device)
                # 訓練時に勾配計算を可能にするためのおまじない→require_gradsがTrueになる
                with torch.set_grad_enabled(model.training):
                    for outer_batch in meta_loader:
                        mean_outer_loss += fmodel(outer_batch, device=device)
                        # x = input('outerbatchが1周しました。続けますかy/n')
                        # if x == 'n':
                            # sys.exit()

                mean_outer_loss.div_(len(outer_batch))
                # 

                # final_loss = inner_loss + mean_outer_loss
                final_loss = inner_loss
                final_loss.backward()
                """


    # Analysis
    model.eval()
    print(model.get_acc(train_loader, device, verbose=True))
    show_acc_loss(numerical_settings, [train_acc, meta_acc, test_acc], loss_list,
                  f'../1toN_result/withoutOuter_BART_metalearning_{ratio}%shot_result_{children_num}.png',
                  labels=['Train(seen all 1to1 & few 1toN)', 'Dev(meta: seen 1toN)', 'Test(unseen 1toN)'])


if __name__ == '__main__':
    main()
