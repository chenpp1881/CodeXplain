import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel
import logging
import json

logger = logging.getLogger(__name__)


class CodeDataset(Dataset):
    def __init__(self, codes, descriptions, labels, tokenizer, args):
        self.codes = codes
        self.descriptions = descriptions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = args.max_len

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        code = self.codes[idx]
        desc = self.descriptions[idx]
        label = self.labels[idx]

        # Tokenize code
        code_tokens = self.tokenizer(code, padding='max_length', max_length=self.max_len, truncation=True,
                                     return_tensors="pt")

        # Tokenize descriptions
        desc_tokens = self.tokenizer(desc, padding='max_length', max_length=self.max_len, truncation=True,
                                     return_tensors="pt")

        # Convert to single tensors
        code_tokens = {key: val.squeeze(0) for key, val in code_tokens.items()}
        desc_tokens = {key: val.squeeze(0) for key, val in desc_tokens.items()}

        return code_tokens, desc_tokens, torch.tensor(label, dtype=torch.long)


def load_data(args, stage):
    all_label = []
    all_code = []
    all_ex = []
    if stage == 'train':
        path = r'../OurMethod/Data/train.json'
    elif stage == 'test':
        path = r'../OurMethod/Data/test.json'
    else:
        path = r'../OurMethod/Data/validation.json'
    with open(path, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    # iter dic
    for file_id, file in datas.items():
        for contract_id, contract in file.items():
            all_code.append(contract['code'])
            all_label.append(contract['lable'])
            all_ex.append(contract['explanations'])

    return CodeDataset(all_code, all_ex, all_label, T5Tokenizer.from_pretrained(args.model_path), args.max_length)
