import torch
import tqdm
import wandb

from data_nn import mix_dataset
from data_nn import DataProcessor
from model_nn import MLPNet
from utility_nn import get_args_cls, fix_seed
from analysis_nn import show_acc_loss
from train_nn import train


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
    seed = args.seed
    w_and_b = args.wandb
    save_flag = args.save_flag
    fix_seed(seed)
    numerical_settings = {'relation_num': num_relation,
                          'father_num': father_num,
                          'batch_size': batch_size,
                          'epoch_num': epoch_num,
                          'lr': lr,
                          'seed': seed}
    if w_and_b:
        wandb.login()
        wandb.init(project="exp-MLP-1toN-classification-union", entity="nagasawa_h")
        wandb.config = {
            "learning_rate": lr,
            "epochs": epoch_num,
            "batch_size": batch_size,
            '#subject': father_num,
        }
    print(numerical_settings)
    print(f'Device: {device}')

    files = ['data/1to2Relation_with_id_revised.csv',
             'data/1to3Relation_with_id.csv',
             'data/1to4Relation_with_id.csv']

    org_child_pairs, parent_labels, _org_masked_sentences = mix_dataset(files, numerical_settings, one_token=True)

    _child_pairs = [eval(child_pair) for child_pair in org_child_pairs]

    relation_set = ['brother', 'sister', 'friend']
    print(relation_set, 'none')
    operations = ['element', 'set', 'union']
    relational_data = DataProcessor(parent_labels, _child_pairs, relation_set, operations, union=True, seed=42)
    dataloaders = relational_data(batch_size)
    union_dataloader = relational_data.prepare_union_dataloader(['brother', 'sister'], batch_size=8)
    print(relational_data.relation_dict)
    for rel in dataloaders.keys():
        print(rel)
        print(dataloaders[rel].dataset.get_example(1))
    print(dataloaders.keys())
    object_dict = relational_data.object_dict

    model = MLPNet(len(object_dict), len(parent_labels), len(relation_set)+1, len(operations))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    train_1to1_loader = dataloaders['1to1_all_relations']
    eval_1to1_loader = dataloaders['evaluation']

    loss_list = list()
    acc_list_1to1 = list()
    acc_1toNs = dict()
    for rel in relation_set:
        acc_1toNs[rel] = list()
    acc_1toNs['union'] = list()

    # training for element teaching
    for _epoch in tqdm.tqdm(range(epoch_num)):
        loss = train(model, criterion, optimizer, train_1to1_loader, device, mode='1to1')
        loss_list.append(loss)
        model.eval()
        acc_1to1 = model.get_acc(mode='1to1', device=device, data_loader=eval_1to1_loader)
        acc_list_1to1.append(acc_1to1)
        if w_and_b:
            wandb.log({"loss": loss, "acc_1to1": acc_1to1})
        for rel in relation_set:
            acc_1toN = model.get_acc(mode='1toN', device=device, data_loader=dataloaders['1toN'+rel])
            acc_1toNs[rel].append(acc_1toN)
            if w_and_b:
                wandb.log({f"acc_{rel}(1toN)": acc_1toN})
        union_acc = model.get_acc(mode='1toN', device=device, data_loader=union_dataloader)
        acc_1toNs['union'].append(union_acc)
        if w_and_b:
            wandb.log({"acc_union": union_acc})

    # training for set teaching
    criterion = torch.nn.BCELoss()
    train_1toN_loader = dataloaders['1toN_all_relations']
    for _epoch in tqdm.tqdm(range(epoch_num)):
        loss = train(model, criterion, optimizer, train_1toN_loader, device, mode='1toN')
        loss_list.append(loss)
        model.eval()
        acc_1to1 = model.get_acc(mode='1to1', device=device, data_loader=eval_1to1_loader)
        acc_list_1to1.append(acc_1to1)
        if w_and_b:
            wandb.log({"loss": loss, "acc_1to1": acc_1to1})
        for rel in relation_set:
            acc_1toN = model.get_acc(mode='1toN', device=device, data_loader=dataloaders['1toN' + rel])
            acc_1toNs[rel].append(acc_1toN)
            if w_and_b:
                wandb.log({f"acc_{rel}(1toN)": acc_1toN})
        union_acc = model.get_acc(mode='1toN', device=device, data_loader=union_dataloader)
        acc_1toNs['union'].append(union_acc)
        if w_and_b:
            wandb.log({"acc_union": union_acc})

    model.get_acc(mode='1toN', device=device, data_loader=union_dataloader, verbose=True)
    accs = list()
    accs.append(acc_list_1to1)
    for acc in acc_1toNs.values():
        accs.append(acc)

    labels = ['1to1(all instances)', '1toN(brothers)', '1toN(sisters)', '1toN(friends)', 'UNION(bro+sis)']

    title = f'../1toN_result/dataloader_fixed_UNION_CLS_MLP_L=2_id_result_{father_num * 3}_' \
            f'batchsize={batch_size}_lr={lr}_seed={seed}.png'

    show_acc_loss(numerical_settings, accs, loss_list, title, labels=labels)

    if w_and_b:
        wandb.finish()
    if save_flag:
        torch.save(model.state_dict(), f'../models/UNION_CLS_MLP_L=2_{father_num * 3}.pth')


if __name__ == '__main__':
    main()
