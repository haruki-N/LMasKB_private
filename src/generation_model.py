import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utility import fix_seed, flatten_list, float_to_str


class MyBart(torch.nn.Module):
    def __init__(self):
        fix_seed(42)
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    def forward(self, inputs, device='cpu'):
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = inputs['labels'].to(device)
        loss = self.bart(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        return loss

    def estimate(self, inputs, device):
        generated_ids = self.bart.generate(inputs['input_ids'].to(device), max_length=26)
        return generated_ids

    def get_acc(self, data_loader, device, verbose=False, plot=False):
        acc = .0
        wrong_count = 0
        correct_count = 0
        wrong_due2_less = 0
        wrong_due2_more = 0
        wrong_due2_mistake_but_same_length = 0
        self.bart.eval()
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                generated_ids = self.bart.generate(input_ids, num_beams=1).cpu()
                for org_gold, org_pred in zip(batch['labels'], generated_ids[:, 1:]):
                    gold = org_gold[(org_gold != 0) & (org_gold != 1) & (org_gold != 2)]   # <s></s><pad>tokenの部分を無視
                    pred = org_pred[(org_pred != 0) & (org_pred != 1) & (org_pred != 2)]
                    gold_txt = [word.strip(',.') for word in self.tokenizer.decode(gold).split()]
                    pred_txt = [word.strip(',.') for word in self.tokenizer.decode(pred).split()]
                    if set(gold_txt) == set(pred_txt):
                        acc += 1
                        if verbose and correct_count <= 5:
                            print(f'Correct! : {gold_txt}')
                            correct_count += 1
                    else:
                        if len(set(gold_txt)) == len(set(pred_txt)):
                            wrong_due2_mistake_but_same_length += 1
                        elif len(set(gold_txt)) > len(set(pred_txt)):
                            wrong_due2_less += 1
                        else:
                            wrong_due2_more += 1
                        if verbose and wrong_count <= 5:
                            print(f'Correct: {set(gold_txt)}')
                            print(f'Wrong  : {set(pred_txt)}')
                            print('-------------------')
                            wrong_count += 1
                        if plot:
                            title = f'../result/BART_WrongPred_{wrong_count}.png'
                            self.plot_prob_scores(org_pred, ', '.join(gold_txt), title, device)

        if verbose:
            print('WRONG ANALYSIS: ---------')
            print(f'\t more: {wrong_due2_more}')
            print(f'\t less: {wrong_due2_less}')
            print(f'\t name mistake: {wrong_due2_mistake_but_same_length}')
            print('-------------------------')
        return acc / len(data_loader.dataset)

    def plot_prob_scores(self, input_ids, gold, title, device):
        """
            make the table of generation scores with top5 words
        :param input_ids:
        :param gold: str
            like 'O_1, O_2, O_3'
        :param title: str
        :param device:
        :return:
        """
        print(f'Plotting prediction scores of {gold}')
        gen_out = self.bart.generate(input_ids.reshape(1, -1).to(device), num_beams=1, return_dict_in_generate=True,
                                     output_scores=True)
        softmax = torch.nn.Softmax(dim=-1)
        pred_score = torch.stack([score.reshape(-1) for score in gen_out['scores']])
        preds = []
        probs = []
        for word_pred in pred_score:
            scores, idxes = torch.topk(softmax(word_pred.reshape(-1)), 5)
            probs.append(scores.tolist())
            words = self.tokenizer.convert_ids_to_tokens(idxes.tolist())
            pred = [word + ' (' + float_to_str(score * 100)[:4] + '%)' for score, word in zip(scores.tolist(), words)]
            preds.append(pred)

        df = pd.DataFrame(preds[1:-1])  # <s>と</s>の分は除外
        probs = probs[1:-1]
        gold_labels = [child.strip(',') for child in gold.split(' ')]

        # bartはサブワード分割なので、gold_labelもサブワード分割しておく
        gold_labels = [self.tokenizer.convert_ids_to_tokens(self.tokenizer(child)['input_ids'])
                       for child in gold_labels]
        gold_labels = list(flatten_list(gold_labels))
        gold_labels = [subword for subword in gold_labels if subword != '<s>' and subword != '</s>']

        color = np.full_like(df.values, "", dtype=object)
        correct = np.zeros_like(df.values)
        for i in range(len(df.values)):
            for j in range(len(df.values.T)):
                if any([gold in df.values[i, j] for gold in gold_labels]):
                    color[i, j] = (255 / 255, 153 / 255, 0, probs[i][j])
                    correct[i, j] = 1
                else:
                    color[i, j] = 'white'

        fig = plt.figure(figsize=(15, len(df.values)))
        ax1 = fig.add_subplot(111)

        ax1.axis('off')
        rows = [f'word{i + 1}' for i in range(len(df.index))]
        table = ax1.table(cellText=df.values,
                          colLabels=['pred1', 'pred2', 'pred3', 'pred4', 'pred5'],
                          rowLabels=rows,
                          loc="center",
                          cellColours=color,
                          rowColours=["#dcdcdc"] * len(rows),
                          colColours=["#dcdcdc"] * 5)
        table.set_fontsize(18)
        data = df.values
        for i in range(len(df.values)):
            for j in range(len(df.values.T)):
                if correct[i, j] == 1:
                    text = table[i + 1, j].get_text()  # columnsの分をskipするためのi+1
                    text.set_weight('bold')
                    text.set_fontstyle('italic')
                    text.set_color('#008080')
        for pos, cell in table.get_celld().items():  # cellの高さを調節
            if pos[0] == 0:  # columns cell
                cell.set_height((1 / len(df.values)) * 0.4)
            else:
                cell.set_height((1 / len(df.values)) * 0.7)

        fig.tight_layout()

        plt.title(f'Correct Subwords: {gold_labels}', fontsize=20)
        plt.subplots_adjust(top=0.8)
        plt.savefig(title)
        plt.close()

        return None


class MyDistillBart(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("valhalla/distilbart-mnli-12-1")
        self.bart = AutoModelForSequenceClassification.from_pretrained("valhalla/distilbart-mnli-12-1")

        fix_seed(42)

    def forward(self, inputs, device='cpu'):
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = inputs['labels'].to(device)
        loss = self.bart(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        return loss

    def estimate(self, inputs, device):
        generated_ids = self.bart.generate(inputs['input_ids'].to(device), max_length=26)
        return generated_ids

    def get_acc(self, data_loader, device, verbose=False, plot=False):
        acc = .0
        count = 0
        self.bart.eval()
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                generated_ids = self.bart.generate(input_ids, num_beams=1).cpu()
                for org_gold, org_pred in zip(batch['labels'], generated_ids[:, 1:]):
                    gold = org_gold[(org_gold != 0) & (org_gold != 1) & (org_gold != 2)]   # <s></s><pad>tokenの部分を無視
                    pred = org_pred[(org_pred != 0) & (org_pred != 1) & (org_pred != 2)]
                    gold_txt = [word.strip(',.') for word in self.tokenizer.decode(gold).split()]
                    pred_txt = [word.strip(',.') for word in self.tokenizer.decode(pred).split()]
                    if set(gold_txt) == set(pred_txt):
                        acc += 1
                    else:
                        if verbose:
                            if count >= 10:
                                break
                            print('------------------')
                            print(f'Correct: {set(gold_txt)}')
                            print(f'Wrong  : {set(pred_txt)}')
                            count += 1
                        if plot:
                            title = f'../result/BART_WrongPred_{count}.png'
                            self.plot_prob_scores(org_pred, ', '.join(gold_txt), title, device)

        return acc / len(data_loader.dataset)

    def plot_prob_scores(self, input_ids, gold, title, device):
        """
            make the table of generation scores with top5 words
        :param input_ids:
        :param gold: str
            like 'O_1, O_2, O_3'
        :param title: str
        :param device:
        :return:
        """
        print(f'Plotting prediction scores of {gold}')
        gen_out = self.bart.generate(input_ids.reshape(1, -1).to(device), num_beams=1, return_dict_in_generate=True,
                                     output_scores=True)
        softmax = torch.nn.Softmax(dim=-1)
        pred_score = torch.stack([score.reshape(-1) for score in gen_out['scores']])
        preds = []
        probs = []
        for word_pred in pred_score:
            scores, idxes = torch.topk(softmax(word_pred.reshape(-1)), 5)
            probs.append(scores.tolist())
            words = self.tokenizer.convert_ids_to_tokens(idxes.tolist())
            pred = [word + ' (' + float_to_str(score * 100)[:4] + '%)' for score, word in zip(scores.tolist(), words)]
            preds.append(pred)

        df = pd.DataFrame(preds[1:-1])  # <s>と</s>の分は除外
        probs = probs[1:-1]
        gold_labels = [child.strip(',') for child in gold.split(' ')]

        # bartはサブワード分割なので、gold_labelもサブワード分割しておく
        gold_labels = [self.tokenizer.convert_ids_to_tokens(self.tokenizer(child)['input_ids'])
                       for child in gold_labels]
        gold_labels = list(flatten_list(gold_labels))
        gold_labels = [subword for subword in gold_labels if subword != '<s>' and subword != '</s>']

        color = np.full_like(df.values, "", dtype=object)
        correct = np.zeros_like(df.values)
        for i in range(len(df.values)):
            for j in range(len(df.values.T)):
                if any([gold in df.values[i, j] for gold in gold_labels]):
                    color[i, j] = (255 / 255, 153 / 255, 0, probs[i][j])
                    correct[i, j] = 1
                else:
                    color[i, j] = 'white'

        fig = plt.figure(figsize=(15, len(df.values)))
        ax1 = fig.add_subplot(111)

        ax1.axis('off')
        rows = [f'word{i + 1}' for i in range(len(df.index))]
        table = ax1.table(cellText=df.values,
                          colLabels=['pred1', 'pred2', 'pred3', 'pred4', 'pred5'],
                          rowLabels=rows,
                          loc="center",
                          cellColours=color,
                          rowColours=["#dcdcdc"] * len(rows),
                          colColours=["#dcdcdc"] * 5)
        table.set_fontsize(18)
        data = df.values
        for i in range(len(df.values)):
            for j in range(len(df.values.T)):
                if correct[i, j] == 1:
                    text = table[i + 1, j].get_text()  # columnsの分をskipするためのi+1
                    text.set_weight('bold')
                    text.set_fontstyle('italic')
                    text.set_color('#008080')
        for pos, cell in table.get_celld().items():  # cellの高さを調節
            if pos[0] == 0:  # columns cell
                cell.set_height((1 / len(df.values)) * 0.4)
            else:
                cell.set_height((1 / len(df.values)) * 0.7)

        fig.tight_layout()

        plt.title(f'Correct Subwords: {gold_labels}', fontsize=20)
        plt.subplots_adjust(top=0.8)
        plt.savefig(title)
        plt.close()

        return None