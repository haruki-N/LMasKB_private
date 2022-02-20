import torch
from transformers import RobertaModel, RobertaTokenizer

from utility import identify_outliers


class MyRoberta(torch.nn.Module):
    def __init__(self, candidates, forward_func='none', rand_init=False):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        if rand_init:
            self.roberta.init_weights()   # 重みの初期化
        self.fc = torch.nn.Linear(768, candidates)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.forward_func = forward_func

    def forward(self, inputs, device='cpu'):
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        outputs = self.roberta(input_ids, attention_mask)['pooler_output']
        outputs = self.fc(outputs)

        if self.forward_func == 'sigmoid':
            return self.sigmoid(outputs)
        elif self.forward_func == 'softmax':   # lossにtorch.CrossEntropyLoss使っている場合は二重になっちゃうから注意
            return self.softmax(outputs)
        else:
            return outputs

    def get_acc(self, data_loader, device, mode='1to1', verbose=False):
        self.eval()
        acc = .0
        count = 0
        for batch in data_loader:
            with torch.no_grad():
                preds = self(batch, device).cpu()
                for answer, pred in zip(batch['labels'], preds):
                    org_pred = pred
                    pred_idx_1to1 = torch.argmax(pred.cpu()).item()
                    # if answer_idx == pred_idx:
                    if mode == '1to1':
                        if answer == pred_idx_1to1:
                            acc += 1
                        elif verbose and count < 2:
                            count += 1
                            print(f'ANSWER: {answer}')
                            print('')
                            print(f'PRED_idx: {pred_idx_1to1}, {torch.max(pred.cpu())}')
                            print(f'PRED  : {pred}')

                    else:   # evaluation for 1toN relations
                        if self.forward_func == 'sigmoid':
                            pred = torch.round(pred).int()
                            zero_one_pred = pred
                        elif self.forward_func == 'softmax':
                            outlier_idx, _ = identify_outliers(pred.tolist(), mode='std')
                            zero_one_pred = torch.zeros_like(answer)
                            for i in outlier_idx:
                                zero_one_pred[i] = 1
                        else:
                            softmax = torch.nn.Softmax(dim=-1)
                            pred = softmax(pred)
                            outlier_idx, _ = identify_outliers(pred.tolist(), mode='std')
                            zero_one_pred = torch.zeros_like(answer)
                            for i in outlier_idx:
                                zero_one_pred[i] = 1
                        if self.forward_func == 'sigmoid' and torch.equal(answer.int(), pred.int()):
                            acc += 1

                        elif self.forward_func == 'softmax' and torch.equal(answer.int(), zero_one_pred.int()):
                            acc += 1
                        elif torch.equal(answer.int(), zero_one_pred.int()):
                            acc += 1

                        elif verbose and count < 2:
                            count += 1
                            print(f'ANSWER_idx: {torch.nonzero(answer).view(-1).tolist()}')
                            print(f'ANSWER    : {answer}')
                            if self.forward_func == 'sigmoid':
                                print(f'PRED_idx: {torch.nonzero(pred).view(-1).tolist()}')
                                print(f'PRED: {org_pred}')
                                print(f'round(PRED)  : {pred}')
                            else:
                                print(f'PRED_idx: {torch.nonzero(zero_one_pred).view(-1).tolist()}')
                                print(f'PRED: {org_pred}')
                                print(f'zero_one_pred  : {zero_one_pred}')

        return acc / len(data_loader.dataset)

    def get_acc_for_same_input(self, dataloader, device):
        self.eval()
        acc = .0
        for batch in dataloader:
            with torch.no_grad():
                preds = self(batch, device).cpu()
                for answer, pred in zip(batch['labels'], preds):
                    pred_idx = torch.argmax(pred)
                    ans_idxes = (answer == 1).nonzero(as_tuple=True)[0].tolist()

                    if pred_idx in ans_idxes:
                        acc += 1

        return acc / len(dataloader.dataset)

    def get_detail_acc(self, input_list, answer_pairs, distractor_pairs, object2idx, device, verbose=False):
        self.eval()
        softmax = torch.nn.Softmax(dim=-1)
        perfect_match = .0   # 完全正当
        wrong_with_dist = .0   # distractorを含んでいるために誤答
        wrong_with_short = .0   # distractorは含んでいないが、取りこぼしがあるために誤答

        for input_sentence, answer_pair, distractor_pair in zip(input_list, answer_pairs, distractor_pairs):
            with torch.no_grad():
                tokenized = self.tokenizer(input_sentence, return_tensors='pt')
                pred = self(tokenized, device)
                if self.forward_func != 'softmax':
                    pred = softmax(pred)
                outlier_idx, top_N = identify_outliers(pred.view(-1).tolist(), mode='std')
                pred_idxes = set(outlier_idx.tolist())
                gold_idxes = set([object2idx[obje] for obje in answer_pair])
                distractor_idxes = set([object2idx[obje] for obje in distractor_pair])
                answer_in_pred = pred_idxes - distractor_idxes
                dist_in_pred = pred_idxes - gold_idxes

                if pred_idxes == gold_idxes:
                    perfect_match += 1
                elif len(dist_in_pred) > 0:
                    wrong_with_dist += 1
                elif len(dist_in_pred) == 0 and len(answer_in_pred) < len(gold_idxes):
                    wrong_with_short += 1

                if verbose and pred_idxes == gold_idxes:
                    print('CORRECT')
                    print(f'pred_idx: {pred_idxes}')
                    print(f'answer_idx: {gold_idxes}')
                    print(f'distractor_idx: {distractor_idxes}')
                    print('---------------')
                elif verbose:
                    print('WRONG')
                    print(f'pred_idx: {pred_idxes}')
                    print(f'answer_idx: {gold_idxes}')
                    print(f'distractor_idx: {distractor_idxes}')
                    print('---------------')

        return perfect_match / len(input_list), wrong_with_short / len(input_list), wrong_with_dist / len(input_list)

