import torch
import tqdm
import pandas as pd
from transformers import RobertaTokenizerFast
from analysis import show_acc_loss
from data import (
    mix_dataset,
    DataProcessor
)
from utility import (
    get_args_cls, fix_seed
)
from classification_model import MyRoberta


# main
def main():
    args = get_args_cls()

    device = torch.device(f'cuda:{args.gpu_number}') if torch.cuda.is_available() else torch.device('cpu')

    # numerical settings
    num_relation = args.relation
    father_num = args.father
    batch_size = args.batch_size
    epoch_num = args.epoch_num
    lr = args.lr
    rand_init = args.rand_init
    numbering = args.numbering
    seed = args.seed
    fix_seed(seed)
    numerical_settings = {'relation_num': num_relation,
                          'father_num': father_num,
                          'numbering': numbering,
                          'batch_size': batch_size,
                          'epoch_num': epoch_num,
                          'lr': lr,
                          'rand_init': rand_init,
                          'seed': seed}
    print(numerical_settings)
    print(f'Device: {device}')

    files = ['data/1to2Relation_with_id_revised.csv',
             'data/1to3Relation_with_id.csv',
             'data/1to4Relation_with_id.csv']

    org_child_pairs, parent_labels, _org_masked_sentences = mix_dataset(files, numerical_settings, one_token=True)

    _child_pairs = [eval(child_pair) for child_pair in org_child_pairs]
    relation_set = ['child', 'partner', 'aunt', 'uncle', 'cousin', 'friend',
                    'doctor', 'mentor', 'neighbor', 'coworker', 'brother', 'sister', 'teacher'][:num_relation]

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    relational_data = DataProcessor(parent_labels, _child_pairs, relation_set, tokenizer, seed=seed)
    dataloaders = relational_data(batch_size)
    print(dataloaders.keys())
    object_dict = relational_data.object_dict
    print(f'#Objects={len(object_dict)}', len(set(object_dict.values())))

    model = MyRoberta(len(object_dict), forward_func='none', rand_init=rand_init)
    # model_dict = torch.load(f'../models/CLS_MLP_L=2_{num_relation}_{father_num*3}.pth')
    # weight = model_dict['final_fc.weight']
    # bias = model_dict['final_fc.bias']
    # params_for_fc = model.state_dict()
    # params_for_fc['fc.weight'] = weight
    # params_for_fc['fc.bias'] = bias
    # model.load_state_dict(params_for_fc)
    model.to(device)

    for name, param in model.named_parameters():
        if 'roberta' in name:
            param.requires_grad = False

    # for name, param in model.named_parameters():
    #     if name in ['fc.weight', 'fc.bias']:
    #         param.requires_grad = False

    # print('check if the encoder params are freezed')
    # freezed_param = []
    # params = []
    # for name, param in model.named_parameters():
    #     params.append(name)
    #     if not param.requires_grad:
    #         freezed_param.append(name)
    # print(f'Non-freezed: {set(params) - set(freezed_param)}')

    one2one_data_loader = dataloaders['1to1_all_relations']
    eval_loader = dataloaders['evaluation']

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = torch.nn.CrossEntropyLoss()

    loss_list = []
    acc_list_1to1 = []
    acc_1toNs = dict()
    for rel in relation_set:
        acc_1toNs[rel] = list()

    for _epoch in tqdm.tqdm(range(100)):
        model.train()
        running_loss = .0
        for batch in one2one_data_loader:
            optimizer.zero_grad()
            pred = model(batch, device)
            loss = criterion(pred, batch['labels'].reshape(-1).to(device))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        loss_list.append(running_loss / len(one2one_data_loader))
        model.eval()
        # acc_list_1to1.append(model.get_acc(one2one_data_loader, device))
        acc_list_1to1.append(model.get_acc_for_same_input(eval_loader, device))
        for relation in relation_set:
            acc_1toNs[relation].append(model.get_acc(dataloaders['1toN'+relation], device, mode='1toN'))

    for name, param in model.named_parameters():
        if 'roberta' in name:
            param.requires_grad = True

    for _epoch in tqdm.tqdm(range(epoch_num)):
        model.train()
        running_loss = .0
        for batch in one2one_data_loader:
            optimizer.zero_grad()
            pred = model(batch, device)
            loss = criterion(pred, batch['labels'].reshape(-1).to(device))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        loss_list.append(running_loss / len(one2one_data_loader))
        model.eval()
        # acc_list_1to1.append(model.get_acc(one2one_data_loader, device))
        acc_list_1to1.append(model.get_acc_for_same_input(eval_loader, device))
        for relation in relation_set:
            acc_1toNs[relation].append(model.get_acc(dataloaders['1toN'+relation], device, mode='1toN'))

    # Analysis
    model.eval()
    print(f'Acc 1: {max(acc_list_1to1)}')
    accs = list()
    accs.append(acc_list_1to1)
    for acc in acc_1toNs.values():
        accs.append(acc)

    labels = ['1to1(all instances)'] + [f'1toN({rel})' for rel in relation_set]

    title = f'../1toN_result/LM_FREEZED(first100epoch)_CLS_RoBERTa_id_result_Rel={num_relation}_{father_num*3}_' \
            f'batchsize={batch_size}_lr={lr}_seed={seed}.png'
    if numbering:
        title = f'../1toN_result/CLS_RoBERTa_id_result_numbering_Rel={num_relation}_{father_num*3}_{model.forward_func}.png'
    if rand_init:
        title = title.replace('CLS', 'CLS_RandInit')
    show_acc_loss(numerical_settings, accs, loss_list, title, labels=labels)
    df = pd.DataFrame(accs, labels)
    df.to_csv(title.replace('png', 'csv'))


if __name__ == '__main__':
    main()
