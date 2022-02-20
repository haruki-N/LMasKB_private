import random
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers.activations import ACT2FN
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import BartConfig, BartPretrainedModel
from transformers.utils import logging

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput
    )
from transformers.models.bart.modeling_bart import (
    shift_tokens_right,
    _expand_mask,
    _make_causal_mask,
    BartAttention,
    Seq2SeqModelOutput,
    BaseModelOutput,
    BartEncoder
)

from utility import fix_seed, flatten_list, float_to_str


logger = logging.get_logger(__name__)


# ========== BART Model ==========
class BartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # nn.Embedding(vocab数, embedの次元数, pad_idx)
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.custom_vocab_size = 9999   # monkey patch to custom here!!!! → Override at _constrained_main.py
        self.decoder_embed = nn.Embedding(self.custom_vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)
        self.entity_decoder = BartDecoder(config, self.decoder_embed)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.entity_decoder
        # return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        # ここをself.decoder → self.entity_decoder
        decoder_outputs = self.entity_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


# ========== simple seq2seq models ==========
class BARTGenerate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

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


# ========== Decoder constrainced BART Model ==========
class BARTDecoderConstrained(torch.nn.Module):
    def __init__(self, tokenizer, decoder_vocab_size):
        super().__init__()
        self.bart = OreOreBartForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.decoder_tokenizer = tokenizer
        self.encoder_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

        # monkey patch
        self.bart.lm_head = nn.Linear(768, decoder_vocab_size, bias=False)
        self.bart.model.custom_vocab_size = decoder_vocab_size
        self.bart.model.decoder.embed_tokens = self.bart.model.decoder_embed
        padding_idx = self.bart.model.config.pad_token_id
        token_embed = nn.Embedding(decoder_vocab_size, self.bart.model.config.d_model, padding_idx)
        self.bart.model.entity_decoder = BartDecoder(self.bart.model.config, token_embed)

        fix_seed(42)

    def forward(self, inputs, device='cpu'):
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = inputs['labels'].to(device)
        loss = self.bart(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        return loss

    def estimate(self, inputs, device):
        generated_ids = self.bart.generate(inputs['input_ids'].to(device), num_beams=1)
        return generated_ids

    def get_acc(self, data_loader, device, verbose=False, one2one=False, plot=False):
        acc = .0
        wrong_due2_more = 0
        wrong_due2_less = 0
        wrong_due2_mistake_but_same_length = 0
        self.bart.eval()
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                generated_ids = self.bart.generate(input_ids, num_beams=1)
                for gold, pred in zip(batch['labels'], generated_ids[:, 1:]):
                    gold = gold[gold != 1].to(device)   # <pad>tokenの部分を無視
                    pred = pred[pred != 1]

                    # 子どもの名前の集合の一致不一致での正答率
                    gold_set = set(flatten_list(gold.tolist()))
                    pred_set = set(flatten_list(pred.tolist()))
                    pred_set -= {0, 2} # <s> </s> tokenを除外
                    gold_set -= {0, 2}
                    if one2one:   # 正解の子ども(gold_set)のうち、1人を答えられていれば正解(問題文が同じ場合など e.g. firstとかを使わない文)
                        if len(pred_set) > 0 and pred_set <= gold_set:
                            acc += 1

                    elif gold_set == pred_set:
                        acc += 1
                    elif verbose:   # 誤答かつそれを出力
                        if len(pred_set) == len(gold_set):
                            wrong_due2_mistake_but_same_length += 1
                        elif len(pred_set) > len(gold_set):
                            wrong_due2_more += 1
                        else:
                            wrong_due2_less += 1

                        print(f'CorrectSet: {self.decoder_tokenizer.decode(list(gold_set))}')
                        print(f'WrongSet  : {self.decoder_tokenizer.decode(list(pred_set))}')
                        print('==============')

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
            words = self.decoder_tokenizer.decode(idxes.tolist()).split(' ')
            pred = [word + ' (' + float_to_str(score * 100)[:4] + '%)' for score, word in zip(scores.tolist(), words)]
            preds.append(pred)

        df = pd.DataFrame(preds[1:-1])   # <s>と</s>の分は除外
        probs = probs[1:-1]
        gold_labels = [child.strip(',') for child in gold.split(' ')]
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
        rows = [f'word{i+1}' for i in range(len(df.index))]
        table = ax1.table(cellText=df.values,
                          colLabels=['pred1', 'pred2', 'pred3', 'pred4', 'pred5'],
                          rowLabels=rows,
                          loc="center",
                          cellColours=color,
                          rowColours=["#dcdcdc"]*len(rows),
                          colColours=["#dcdcdc"]*5)
        table.set_fontsize(18)
        data = df.values
        for i in range(len(df.values)):
            for j in range(len(df.values.T)):
                if correct[i, j] == 1:
                    text = table[i+1, j].get_text()   # columnsの分をskipするためのi+1
                    text.set_weight('bold')
                    text.set_fontstyle('italic')
                    text.set_color('#008080')
        for pos, cell in table.get_celld().items():   # cellの高さを調節
            if pos[0] == 0:   # columns cell
                cell.set_height((1/len(df.values))*0.4)
            else:
                cell.set_height((1/len(df.values))*0.7)

        fig.tight_layout()

        plt.title(f'Correct Childrens are: {gold_labels}', fontsize=24)
        plt.subplots_adjust(top=0.8)
        plt.savefig(title)
        plt.close()

        return None


# ========== BART For Conditional Generation ==========
class OreOreBartForConditionalGeneration(BartPretrainedModel):
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        lm_logits = self.lm_head(outputs[0])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.model.custom_vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


# =========== Customized BART Decoder for constrained ==========
class BartDecoder(BartPretrainedModel):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


# ========== BartDecoderLayer ==========
class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):

        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# ========== BartLearnedPositionalEmbedding ==========
class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)
