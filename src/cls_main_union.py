import torch
import tqdm
import random, copy
import pandas as pd
from transformers import RobertaTokenizerFast
from analysis import show_acc_loss, show_prob_distribution_classification, plot_single_fig
from data import (
    mix_dataset,
    prepare_dataloader_cls_1to1,
    prepare_dataloader_cls_1ton,
    prepare_relations
)
from utility import (
    get_args_cls, fix_seed, flatten_list,
)
from classification_model import MyRoberta


# main
def main():
    args = get_args_cls()

    device = torch.device(f'cuda:{args.gpu_number}') if torch.cuda.is_available() else torch.device('cpu')

    # numerical settings
    relation_num = args.relation
    father_num = args.father
    batch_size = args.batch_size
    epoch_num = args.epoch_num
    lr = args.lr
    seed = args.seed
    numbering = args.numbering
    mix_ratio = args.mix_ratio
    fix_seed(seed)
    numerical_settings = {'relation_num': relation_num,
                          'father_num': father_num,
                          'mix_ratio': mix_ratio,
                          'numbering': numbering,
                          'batch_size': batch_size,
                          'epoch_num': epoch_num,
                          'lr': lr,
                          'seed': seed}
    print(numerical_settings)

    files = ['data/1to2Relation_with_id_revised.csv',
             'data/1to3Relation_with_id.csv',
             'data/1to4Relation_with_id.csv']

    org_child_pairs, parent_labels, _org_masked_sentences = mix_dataset(files, numerical_settings, one_token=True)

    child_pairs = [eval(child_pair) for child_pair in org_child_pairs]

    one2one_son_input, one2one_son_labels, one2n_son_input, one2n_son_labels = prepare_relations(parent_labels,
                                                                                                 child_pairs,
                                                                                                 relation='son')
    one2one_dau_input, one2one_dau_labels, one2n_dau_input, one2n_dau_labels = prepare_relations(parent_labels,
                                                                                                 child_pairs,
                                                                                                 relation='daughter')
    # one2one_bro_input, one2one_bro_labels, one2n_bro_input, one2n_bro_labels = prepare_relations(parent_labels,
    #                                                                                              child_pairs,
    #                                                                                              relation='brother')
    # one2one_sis_input, one2one_sis_labels, one2n_sis_input, one2n_sis_labels = prepare_relations(parent_labels,
    #                                                                                              child_pairs,
    #                                                                                              relation='sister')
    one2one_fre_input, one2one_fre_labels, one2n_fre_input, one2n_fre_labels = prepare_relations(parent_labels,
                                                                                                 child_pairs,
                                                                                                 relation='friend')
    li = list(zip(parent_labels, child_pairs))
    random.shuffle(li)
    shuffled_parent_labels, shuffled_child_pairs = zip(*li)
    sample_num = int(father_num * (mix_ratio / 100))
    sampled_subjects = shuffled_parent_labels[:sample_num]
    sampled_pairs = shuffled_child_pairs[:sample_num]

    one2one_child_input, _, _, _ = prepare_relations(sampled_subjects, sampled_pairs, relation='child')
    _, son_as_child_1to1_lables, _, _ = prepare_relations(sampled_subjects, sampled_pairs, relation='son')
    _, daughter_as_child_1to1_lables, _, _ = prepare_relations(sampled_subjects, sampled_pairs, relation='daughter')

    # one2one_sibling_input, _, _, _ = prepare_relations(sampled_subjects, sampled_pairs, relation='sibling')
    # _, bro_as_sibling_1to1_lables, _, _ = prepare_relations(sampled_subjects, sampled_pairs, relation='brother')
    # _, sis_as_sibling_1to1_lables, _, _ = prepare_relations(sampled_subjects, sampled_pairs, relation='sister')

    one2one_train_sentence = one2one_son_input + one2one_dau_input + one2one_fre_input +\
                             one2one_child_input + one2one_child_input
    one2one_train_labels = one2one_son_labels + one2one_dau_labels +one2one_fre_labels +\
                           son_as_child_1to1_lables + daughter_as_child_1to1_lables
    # one2one_train_sentence = one2one_bro_input + one2one_sis_input + one2one_fre_input + \
    #                          one2one_sibling_input + one2one_sibling_input
    # one2one_train_labels = one2one_bro_labels + one2one_sis_labels + one2one_fre_labels + \
    #                        bro_as_sibling_1to1_lables + sis_as_sibling_1to1_lables
    # print(f'#sampled subjects = {sample_num}, #sampled sons = {len(son_as_child_1to1_lables)}, '
    #       f'#sampled daughters = {len(daughter_as_child_1to1_lables)}')
    object_labels = one2one_son_labels + one2one_dau_labels + one2one_fre_labels
    # object_labels = one2one_bro_labels + one2one_sis_labels + one2one_fre_labels
    num_all_objects = len(object_labels)
    object2idx = dict(zip(object_labels, list(range(num_all_objects))))
    print(f'ここチェック：all_obje={num_all_objects}, object2idx={len(object2idx)}, unique_obje={len(set(object_labels))}')
    print('Dataset Prepared')
    print(f'Train: {len(one2one_train_sentence)}, {len(one2one_train_labels)}')
    # print(f'#Fathers = {len(parent_labels)}, #Sons = {len(one2one_son_labels)}, #Daughters = {len(one2one_dau_labels)}')

    # Training Setup
    print(f'Device: {device}')
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    model = MyRoberta(num_all_objects, forward_func='none')
    model.to(device)
    # model_dict = torch.load(f'../models/UNION_CLS_MLP_L=2_{father_num * 3}.pth')
    # weight = model_dict['final_fc.weight']
    # bias = model_dict['final_fc.bias']
    # params_for_fc = model.state_dict()
    # params_for_fc['fc.weight'] = weight
    # params_for_fc['fc.bias'] = bias
    # model.load_state_dict(params_for_fc)
    # model.to(device)
    #
    # for name, param in model.named_parameters():
    #     if name in ['fc.weight', 'fc.bias']:
    #         param.requires_grad = False
    #
    # print('check if the encoder params are freezed')
    # freezed_param = []
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         freezed_param.append(name)
    # print(f'FREEZED: {freezed_param}')

    one2one_son_dau_fre_loader = prepare_dataloader_cls_1to1(one2one_train_sentence, one2one_train_labels, object2idx,
                                                             tokenizer, batch_size, max_length=36)

    one2n_children_labels = []
    one2n_children_sent = [f"Who are the children of {parent}?" for parent in parent_labels]
    for son_pair, dau_pair in zip(one2n_son_labels, one2n_dau_labels):
        one2n_children_labels.append(son_pair+dau_pair)

    # for son_pair, dau_pair in zip(one2n_bro_labels, one2n_sis_labels):
    #     one2n_children_labels.append(son_pair+dau_pair)

    one2n_children_loader = prepare_dataloader_cls_1ton(parent_labels, one2n_children_labels, object2idx,
                                                        num_all_objects, tokenizer, batch_size=batch_size,
                                                        max_length=36, relation='child')
    son_and_dau_input = [f'Who are the brothers and sisters of {parent}?' for parent in parent_labels]
    one2n_son_dau_loader = prepare_dataloader_cls_1ton(son_and_dau_input, one2n_children_labels, object2idx,
                                                       num_all_objects, tokenizer, batch_size=batch_size,
                                                       max_length=36, relation='none')

    one2n_son_loader = prepare_dataloader_cls_1ton(one2n_son_input, one2n_son_labels, object2idx, num_all_objects,
                                                   tokenizer, batch_size=batch_size, relation='none')
    one2n_dau_loader = prepare_dataloader_cls_1ton(one2n_dau_input, one2n_dau_labels, object2idx, num_all_objects,
                                                   tokenizer, batch_size=batch_size, relation='none')
    # one2n_bro_loader = prepare_dataloader_cls_1ton(one2n_bro_input, one2n_bro_labels, object2idx, num_all_objects,
    #                                                tokenizer, batch_size=batch_size, relation='none')
    # one2n_sis_loader = prepare_dataloader_cls_1ton(one2n_sis_input, one2n_sis_labels, object2idx, num_all_objects,
    #                                                tokenizer, batch_size=batch_size, relation='none')
    one2n_fre_loader = prepare_dataloader_cls_1ton(one2n_fre_input, one2n_fre_labels, object2idx, num_all_objects,
                                                   tokenizer, batch_size=batch_size, relation='none')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = torch.nn.CrossEntropyLoss()

    loss_list = []
    acc_list_1to1 = []
    acc_list_1toN_son = []
    acc_list_1toN_dau = []
    acc_list_1toN_fre = []
    acc_list_1toN_children = []
    acc_list_1toN_son_dau = []

    perfect_match = []
    wrong_with_short = []
    wrong_with_distractor = []
    perfect_match_sondau = []
    wrong_with_short_sondau = []
    wrong_with_distractor_sondau = []

    # for eval
    one2one_eval_input = [f'Who is the son of {parent}?' for parent in parent_labels] + \
                         [f'Who is the daughter of {parent}?' for parent in parent_labels]
    one2one_eval_labels = one2n_son_labels + one2n_dau_labels
    # one2one_eval_input = [f'Who is the brother of {parent}?' for parent in parent_labels] + \
    #                      [f'Who is the sister of {parent}?' for parent in parent_labels]
    # one2one_eval_labels = one2n_bro_labels + one2n_sis_labels

    for _epoch in tqdm.tqdm(range(epoch_num)):
        model.train()
        running_loss = .0
        for batch in one2one_son_dau_fre_loader:
            optimizer.zero_grad()
            pred = model(batch, device)
            loss = criterion(pred, batch['labels'].to(device))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        loss_list.append(running_loss / len(one2one_son_dau_fre_loader))
        model.eval()
        acc_list_1to1.append(model.get_acc_for_same_input(one2one_eval_input, one2one_eval_labels, object2idx, device))
        acc_list_1toN_son.append(model.get_acc(one2n_son_loader, device, mode='1toN'))
        acc_list_1toN_dau.append(model.get_acc(one2n_dau_loader, device, mode='1toN'))
        acc_list_1toN_fre.append(model.get_acc(one2n_fre_loader, device, mode='1toN'))
        acc_list_1toN_children.append(model.get_acc(one2n_children_loader, device, mode='1toN'))
        # acc_list_1toN_son_dau.append(model.get_acc(one2n_son_dau_loader, device, mode='1toN'))

        perfect, short, distractor = model.get_detail_acc(one2n_children_sent, one2n_children_labels, one2n_fre_labels,
                                                          object2idx, device)
        perfect_match.append(perfect * 100)
        wrong_with_short.append(short * 100)
        wrong_with_distractor.append(distractor * 100)

        perfect, short, distractor = model.get_detail_acc(son_and_dau_input, one2n_children_labels, one2n_fre_labels,
                                                          object2idx, device)
        perfect_match_sondau.append(perfect * 100)
        wrong_with_short_sondau.append(short * 100)
        wrong_with_distractor_sondau.append(distractor * 100)

        if _epoch % 100 == 0:
            print(acc_list_1to1[-1], acc_list_1toN_son[-1], acc_list_1toN_dau[-1],
                  acc_list_1toN_fre[-1], acc_list_1toN_children[-1])

    _ = model.get_detail_acc(one2n_children_sent, one2n_children_labels, one2n_fre_labels, object2idx, device, verbose=True)
    # Analysis
    accs = [acc_list_1to1, acc_list_1toN_son, acc_list_1toN_dau, acc_list_1toN_fre,
            acc_list_1toN_children]
    labels = ['1to1(son/daughter/friend)', '1toN(sons)', '1toN(daughters)', '1toN(friends)',
              '1toN(children)']
    title = f'../1toN_result/CLS_RoBERTa_id_result_UNION_son_daughter_fre_{father_num*3}_' \
            f'subject{mix_ratio}%shot_batchsize={batch_size}_lr={lr}_seed={seed}_{model.forward_func}.png'
    # labels = ['1to1(bro/sis/friend)', '1toN(brothers)', '1toN(sisters)', '1toN(friends)',
    #           '1toN(siblings)', '1toN(brothers & sisters)']
    # title = f'../1toN_result/CLS_RoBERTa_id_result_UNION_bro_sis_fre_{father_num * 3}_' \
    #         f'sibling{mix_ratio}%shot_batchsize={batch_size}_lr={lr}_seed={seed}_{model.forward_func}.png'
    show_acc_loss(numerical_settings, accs, loss_list, title, labels=labels)
    df = pd.DataFrame(accs, labels)
    df.to_csv(title.replace('png', 'csv'))
    print(max(acc_list_1toN_children))

    # detailed analysis
    # title = f'../1toN_result/CLS_RoBERTa_id_result_UNION_son_daughter_fre_{father_num * 3}_' \
    #         f'{mix_ratio}%shot_batchsize={batch_size}_lr={lr}_seed={seed}_children_detailed_{model.forward_func}.png'
    # plot_single_fig([perfect_match, wrong_with_short, wrong_with_distractor], title,
    #                 'model prediction ratio(ask "children")', ['perfect match', 'lacking gold', 'predict distractor'],
    #                 'epoch', 'ratio(%)')
    # title = f'../1toN_result/CLS_RoBERTa_id_result_UNION_son_daughter_fre_{father_num * 3}_' \
    #         f'{mix_ratio}%shot_batchsize={batch_size}_lr={lr}_seed={seed}_son&dau_detailed_{model.forward_func}.png'
    # plot_single_fig([perfect_match_sondau, wrong_with_short_sondau, wrong_with_distractor_sondau], title,
    #                 'model prediction ratio(ask "sons & daughters")', ['perfect match', 'lacking gold', 'predict distractor'],
    #                 'epoch', 'ratio(%)')

    title = f'../1toN_result/CLS_RoBERTa_id_result_UNION_bro_sis_fre_{father_num * 3}_' \
            f'sibling{mix_ratio}%shot_batchsize={batch_size}_lr={lr}_seed={seed}_sibling_detailed_{model.forward_func}.png'
    plot_single_fig([perfect_match, wrong_with_short, wrong_with_distractor], title,
                    'model prediction ratio(ask "sibling")', ['perfect match', 'lacking gold', 'predict distractor'],
                    'epoch', 'ratio(%)')
    title = f'../1toN_result/CLS_RoBERTa_id_result_UNION_bro_sis_fre_{father_num * 3}_' \
            f'sibling{mix_ratio}%shot_batchsize={batch_size}_lr={lr}_seed={seed}_bro&sis_detailed_{model.forward_func}.png'
    plot_single_fig([perfect_match_sondau, wrong_with_short_sondau, wrong_with_distractor_sondau], title,
                    'model prediction ratio(ask "brothers & sisters")',
                    ['perfect match', 'lacking gold', 'predict distractor'],
                    'epoch', 'ratio(%)')

    # softmax = torch.nn.Softmax(dim=-1)
    #
    # ex_parent = parent_labels[-5]
    # ex_child = child_pairs[-5]
    # sent = f'Who are the sons of {ex_parent}?'
    # tokenized = tokenizer(sent, return_tensors='pt')
    # outputs = model(tokenized, device).view(-1)
    # if model.forward_func == 'none':
    #     outputs = softmax(outputs)
    # title = f'probs. (INPUT: {sent}) processed through softmax'
    # file_name = f'../1toN_result/prob_dist_UNION_son_daughter_fre_ask_sons_1to4_processed_softmax.png'
    # show_prob_distribution_classification(numerical_settings, outputs.tolist(), ex_child, title, file_name)
    #
    # sent = f'Who are the daughters of {ex_parent}?'
    # tokenized = tokenizer(sent, return_tensors='pt')
    # outputs = model(tokenized, device).view(-1)
    # if model.forward_func == 'none':
    #     outputs = softmax(outputs)
    # title = f'probs. (INPUT: {sent}) processed through softmax'
    # file_name = f'../1toN_result/prob_dist_UNION_son_daughter_fre_ask_daughters_1to4_processed_softmax.png'
    # show_prob_distribution_classification(numerical_settings, outputs.tolist(), ex_child, title, file_name)
    #
    # sent = f'Who are the children of {ex_parent}?'
    # tokenized = tokenizer(sent, return_tensors='pt')
    # outputs = model(tokenized, device).view(-1)
    # if model.forward_func == 'none':
    #     outputs = softmax(outputs)
    # title = f'probs. (INPUT: {sent}) processed through softmax'
    # file_name = f'../1toN_result/prob_dist_UNION_son_daughter_fre_ask_children_1to4_processed_softmax.png'
    # show_prob_distribution_classification(numerical_settings, outputs.tolist(), ex_child, title, file_name)
    #
    # sent = f'Who are the sons and daughters of {ex_parent}?'
    # tokenized = tokenizer(sent, return_tensors='pt')
    # outputs = model(tokenized, device).view(-1)
    # if model.forward_func == 'none':
    #     outputs = softmax(outputs)
    # title = f'probs. (INPUT: {sent}) processed through softmax'
    # file_name = f'../1toN_result/prob_dist_UNION_son_daughter_fre_ask_son&dau_1to4_processed_softmax.png'
    # show_prob_distribution_classification(numerical_settings, outputs.tolist(), ex_child, title, file_name)
    #
    # sent = f'Who are the friends of {ex_parent}?'
    # tokenized = tokenizer(sent, return_tensors='pt')
    # outputs = model(tokenized, device).view(-1)
    # if model.forward_func == 'none':
    #     outputs = softmax(outputs)
    # title = f'probs. (INPUT: {sent}) processed through softmax'
    # file_name = f'../1toN_result/prob_dist_UNION_son_daughter_fre_ask_friends_1to4_processed_softmax.png'
    # show_prob_distribution_classification(numerical_settings, outputs.tolist(), ex_child, title, file_name)


if __name__ == '__main__':
    main()
