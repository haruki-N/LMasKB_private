import torch
import torch.nn as nn
import torch.nn.functional as F

from utility_nn import identify_outliers


class MLPNet(nn.Module):
    def __init__(self, output_dim: int, subject_size: int, rel_size: int, operation_size: int, drop_out=False):
        super(MLPNet, self).__init__()
        self.dropout = drop_out
        self.emb_sub_dim = 100
        self.emb_rel_dim = 10
        self.emb_ope_dim = 10
        self.input_dim = self.emb_sub_dim + self.emb_rel_dim*2 + self.emb_ope_dim
        self.fc1 = nn.Linear(self.input_dim, 768)
        self.fc2 = nn.Linear(768, 768)
        self.final_fc = nn.Linear(768, output_dim)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        # prepare embeddings
        self.emb_subject = nn.Embedding(subject_size, self.emb_sub_dim)
        self.emb_relation = nn.Embedding(rel_size, self.emb_rel_dim)
        self.emb_operation = nn.Embedding(operation_size, self.emb_ope_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs, device):
        subject = inputs['subject'].to(device)
        relation1 = inputs['relation1'].to(device)
        relation2 = inputs['relation2'].to(device)
        operation = inputs['operation'].to(device)

        input_list = list()
        input_list.append(self.emb_subject(torch.argmax(subject, dim=1)))
        input_list.append(self.emb_relation(torch.argmax(relation1, dim=1)))
        input_list.append(self.emb_relation(torch.argmax(relation2, dim=1)))
        input_list.append(self.emb_operation(torch.argmax(operation, dim=1)))

        x = torch.cat(input_list, dim=1)

        x = F.relu(self.fc1(x))
        if self.dropout:
            x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        if self.dropout:
            x = self.dropout2(x)

        return self.final_fc(x)

    def get_acc(self, mode='1oN', device='cpu', data_loader=None, verbose=False):
        self.eval()
        acc = .0
        for batch in data_loader:
            with torch.no_grad():
                preds = self(batch, device).cpu()
                if preds.dim() == 1:
                    preds = preds.view(1, -1)
                for answer, pred in zip(batch['answer'], preds):
                    ans_idxes = (answer == 1).nonzero(as_tuple=True)[0].tolist()
                    softmax = torch.nn.Softmax(dim=-1)
                    pred = softmax(pred)
                    top_1 = torch.argmax(pred).item()
                    outlier_idx, _ = identify_outliers(pred.tolist(), mode='std')
                    zero_one_pred = torch.zeros_like(answer)
                    for i in outlier_idx:
                        zero_one_pred[i] = 1

                    if mode == '1toN' and torch.equal(answer.int(), zero_one_pred.int()):
                        acc += 1
                    elif mode == '1to1' and top_1 in ans_idxes:
                        acc += 1

                    elif verbose:
                        # count += 1
                        print(f'ANSWER_idx: {torch.nonzero(answer).view(-1).tolist()}')
                        # print(f'ANSWER    : {answer}')

                        if mode == '1toN':
                            print(f'PRED_idx  : {torch.nonzero(zero_one_pred).view(-1).tolist()}')
                            # print(f'PRED      : {org_pred}')
                        else:
                            print(f'PRED_idx  : {top_1}')
                            # print(f'PRED      : {pred}')

        return acc / len(data_loader.dataset)
