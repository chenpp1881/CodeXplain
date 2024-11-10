import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5EncoderModel
import logging
import json

logger = logging.getLogger(__name__)


class CodeDataset(Dataset):
    def __init__(self, datas, tokenizer, max_lenth):
        self.datas = datas
        self.tokenizer = tokenizer
        self.max_len = max_lenth

    def __len__(self):
        return len(self.datas)

    def load_tokens(self,text):
        text_tokens = self.tokenizer(text, padding='max_length', max_length=self.max_len, truncation=True,
                                return_tensors="pt")
        return {key: val.squeeze(0) for key, val in text_tokens.items()}

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx])
        code = data['code']
        label = 1 if data['label'] == "vulnerable" else 0

        BFI = data['Basic Functionality Interpretation']
        SSA = data['Step-by-Step Analysis']
        CIA = data['Contract Interaction Analysis']
        OAC = data['Ownership and Access Control']
        GEE = data['Gas Efficiency Examination']
        LFI = data['Logic and Flow Interpretation']
        SMA = data['State Management Analysis']
        EFI = data['Event and Function Interaction']
        EHE = data['Error Handling and Exceptions']

        # Tokenize code
        CODE = self.load_tokens(code)
        BFI = self.load_tokens(BFI)
        SSA = self.load_tokens(SSA)
        CIA = self.load_tokens(CIA)
        OAC = self.load_tokens(OAC)
        GEE = self.load_tokens(GEE)
        LFI = self.load_tokens(LFI)
        SMA = self.load_tokens(SMA)
        EFI = self.load_tokens(EFI)
        EHE = self.load_tokens(EHE)

        return CODE, [BFI, SSA, CIA, OAC, GEE, LFI, SMA, EFI, EHE], torch.tensor(label, dtype=torch.long)

def load_data(args, stage):
    if stage == 'train':
        path = r'VD-data/contract_check/Filtered_DS/Solidity_train.jsonl'
    elif stage == 'test':
        path = r'VD-data/contract_check/Filtered_DS/Solidity_test.jsonl'
    else:
        path = r'VD-data/contract_check/Filtered_DS/Solidity_val.jsonl'
    with open(path, 'r', encoding='utf-8') as f:
        datas = f.readlines()

    return CodeDataset(datas, AutoTokenizer.from_pretrained(args.model_path), args.max_length)
