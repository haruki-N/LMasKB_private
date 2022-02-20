import torch
import tqdm
from transformers import RobertaTokenizerFast
from analysis import show_acc_loss, show_prob_distribution_classification
from data import (
    mix_dataset,
    prepare_dataloader_cls_1to1,
    prepare_dataloader_cls_1ton,
    prepare_relations,
    prepare_relations_for_union
)
from utility import (
    get_args_cls, fix_seed, flatten_list,
)
from classification_model import MyRoberta


# main
def main():
    args = get_args_cls()

    device = torch.device(f'cuda:{args.gpu_number}') if torch.cuda.is_available() else torch.device('cpu')
    # 乱数シード値の固定
    fix_seed(42)

    # DataSet
    # numerical settings
    relation_num = args.relation
    father_num = args.father
    batch_size = args.batch_size
    epoch_num = args.epoch_num
    lr = args.lr
    numbering = args.numbering
    numerical_settings = {'relation_num': relation_num,
                          'father_num': father_num,
                          'numbering': numbering,
                          'batch_size': batch_size,
                          'epoch_num': epoch_num,
                          'lr': lr}
    print(numerical_settings)

    files = ['data/1to2Relation_with_id_revised.csv',
             'data/1to3Relation_with_id.csv',
             'data/1to4Relation_with_id.csv']

    org_child_pairs, parent_labels, _org_masked_sentences = mix_dataset(files, numerical_settings, one_token=True)

    child_labels = list(flatten_list([eval(child_pair) for child_pair in org_child_pairs]))
    child_pairs = [eval(child_pair) for child_pair in org_child_pairs]
    input_grandchild_1to1, grandchild_pairs, eval_input, eval_labels, one2n_grandchild_labels = prepare_relations_for_union(parent_labels, child_pairs)

    grandchild_labels = list(flatten_list(grandchild_pairs))
    all_labels = child_labels + grandchild_labels
    # input_child_1to1 = []
    # for parent, children in zip(parent_labels, child_pairs):
        # input_child_1to1.extend([f'Who is the child of {parent}?' for i, _ in enumerate(children)])
    one2one_sentence = input_grandchild_1to1    # + input_child_1to1
    num_all_objects = len(all_labels)
    object2idx = dict(zip(all_labels, list(range(len(all_labels)))))
    print('Dataset Prepared')
    print(f'Train: {len(one2one_sentence)}, {len(grandchild_labels)}')
    print(f'Eval: {len(eval_input)}, {len(eval_labels)}')
    print(f'#Fathers = {len(parent_labels)}, #Children = {len(child_labels)}, #GrandChildren = {len(grandchild_labels)}')

    # Training Setup
    print(f'Device: {device}')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    model = MyRoberta(num_all_objects, forward_func='none')
    model.to(device)

    one2one_data_loader = prepare_dataloader_cls_1to1(one2one_sentence, grandchild_labels, object2idx, tokenizer,
                                                      batch_size, max_length=36)

    one2n_grandchild_loader = prepare_dataloader_cls_1ton(parent_labels, one2n_grandchild_labels, object2idx,
                                                          num_all_objects, tokenizer, batch_size=batch_size,
                                                          max_length=36, relation_type='grandchild')

    one2n_child_loader = prepare_dataloader_cls_1ton(parent_labels, child_pairs, object2idx, num_all_objects,
                                                     tokenizer, batch_size=batch_size, relation_type='child')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = torch.nn.CrossEntropyLoss()

    loss_list = []
    acc_list_1to1 = []
    acc_list_1toN = []
    acc_list_1toN_grandchild = []

    for _epoch in tqdm.tqdm(range(epoch_num)):
        model.train()
        running_loss = .0
        for batch in one2one_data_loader:
            optimizer.zero_grad()
            pred = model(batch, device)
            loss = criterion(pred, batch['labels'].to(device))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        loss_list.append(running_loss / len(one2one_data_loader))
        model.eval()
        # acc_list_1to1.append(model.get_acc(one2one_data_loader, device))
        acc_list_1to1.append(model.get_acc_for_same_input(eval_input, eval_labels, object2idx, device))
        acc_list_1toN.append(model.get_acc(one2n_child_loader, device, mode='1toN'))
        acc_list_1toN_grandchild.append(model.get_acc(one2n_grandchild_loader, device, mode='1toN'))

        if _epoch % 100 == 0:
            print(acc_list_1to1[-1], acc_list_1toN[-1])

    bce = torch.nn.BCELoss()
    model.forward_func = 'softmax'
    for _epoch in tqdm.tqdm(range(epoch_num)):
        model.train()
        running_loss = .0
        for batch in one2n_grandchild_loader:
            optimizer.zero_grad()
            pred = model(batch, device)
            loss = bce(pred, batch['labels'].float().to(device))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        loss_list.append(running_loss / len(one2one_data_loader))
        model.eval()
        # acc_list_1to1.append(model.get_acc(one2one_data_loader, device))
        acc_list_1to1.append(model.get_acc_for_same_input(eval_input, eval_labels, object2idx, device))
        acc_list_1toN.append(model.get_acc(one2n_child_loader, device, mode='1toN'))
        acc_list_1toN_grandchild.append(model.get_acc(one2n_grandchild_loader, device, mode='1toN'))

        if _epoch % 100 == 0:
            print(acc_list_1to1[-1], acc_list_1toN[-1])

    # Analysis
    accs = [acc_list_1to1, acc_list_1toN, acc_list_1toN_grandchild]
    labels = ['1to1(prefix+child → grandchild)', '1toN(parent → children)', '1toN(parent → grandchildren)']
    title = f'../1toN_result/CLS_RoBERTa_id_result_UNION_then_GrandChild1toN{father_num*3}.png'
    show_acc_loss(numerical_settings, accs, loss_list, title, labels=labels)

    ex_parent = parent_labels[-5]
    ex_children = child_pairs[-5]

    inputs = [input_grandchild_1to1[-5]]

    softmax = torch.nn.Softmax(dim=-1)
    for sent in inputs:
        print(sent)
        tokenized = tokenizer(sent, return_tensors='pt')
        outputs = model(tokenized, device).view(-1)
        if model.forward_func == 'none':
            outputs = softmax(outputs)
        title = f'probs. (INPUT: {sent}) processed through softmax'
        file_name = f'../1toN_result/prob_dist_UNION_then_GrandChild_1toNask_1to1_processed_softmax.png'
        show_prob_distribution_classification(numerical_settings, outputs.tolist(), ex_children, title, file_name)

    ex_parent = parent_labels[-5]
    ex_child = child_pairs[-5]
    sent = f'Who are the grandchildren of {ex_parent}?'
    tokenized = tokenizer(sent, return_tensors='pt')
    outputs = model(tokenized, device).view(-1)
    if model.forward_func == 'none':
        outputs = softmax(outputs)
    title = f'probs. (INPUT: {sent}) processed through softmax'
    file_name = f'../1toN_result/prob_dist_UNION_then_GrandChild1toN_ask_grandchild_1to4_processed_softmax.png'
    show_prob_distribution_classification(numerical_settings, outputs.tolist(), ex_child, title, file_name)

    sent = f'Who are the children of {ex_parent}?'
    tokenized = tokenizer(sent, return_tensors='pt')
    outputs = model(tokenized, device).view(-1)
    if model.forward_func == 'none':
        outputs = softmax(outputs)
    title = f'probs. (INPUT: {sent}) processed through softmax'
    file_name = f'../1toN_result/prob_dist_UNION_then_GrandChild1toN_ask_child_1to4_processed_softmax.png'
    show_prob_distribution_classification(numerical_settings, outputs.tolist(), ex_child, title, file_name)


if __name__ == '__main__':
    main()
