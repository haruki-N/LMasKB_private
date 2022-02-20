import os
import json
import torch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


class MyTokenizer():
    def __init__(self, words2ids: dict):
        # make sure that <pad> token's id should be 1
        self.words2ids = words2ids

        # prepare json file from 'words2ids'
        with open('words2ids.json', 'w') as f:
            json.dump(words2ids, f, indent=2)

        # self.tokenizer = Tokenizer(WordLevel(WordLevel.read_file('words2ids.json')))
        self.tokenizer = Tokenizer(WordLevel(self.words2ids, unk_token='<unk>'))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.post_processor = TemplateProcessing(single="<s> $A </s>",
                                        special_tokens=[("<s>", 0), ("</s>", 2), ])

        os.remove('words2ids.json')

    def tokenize(self, texts: list, pad_length=16, check_oov=True):
        tokenized = self.tokenizer.encode_batch(texts)
        # padding
        [output.pad(pad_length, pad_id=1) for output in tokenized]
        input_ids = [torch.tensor(output.ids) for output in tokenized]
        attention_mask = [torch.tensor(output.attention_mask) for output in tokenized]

        # check OOV
        if check_oov:
            print('Checking OOV at constrained Decoder')
            for i, output in enumerate(tokenized):
                if 3 in output.ids:
                    print(f'OOV: {texts[i]}({self.tokenizer.decode(output.ids)})')

            print('Done: Checking OOV')

        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask)
        }

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.reshape(-1).tolist()

        return self.tokenizer.decode(ids)

    def vocab_size(self):
        return len(self.words2ids)
