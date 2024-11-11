import torch
import torch.nn as nn
from Multihead_Attention import DecoderAttention
from SubLayerConnection import SublayerConnection
from transformers import T5EncoderModel
import random
import pdb

class InformationFusionBlock(nn.Module):

    def __init__(self, hidden, dropout, args):
        super().__init__()
        self.args = args
        self.num_layers = 9
        self.self_attention = nn.ModuleList()
        self.sublayer_connection1 = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.codeT5 = T5EncoderModel.from_pretrained(args.model_path)
        self.fc = nn.Linear(hidden, 2)
        for _layer in range(self.num_layers):
            self.self_attention.append(DecoderAttention(d_model=hidden))
            self.sublayer_connection1.append(SublayerConnection(size=hidden, dropout=dropout))
            self.linear_layers.append(nn.Linear(in_features=hidden, out_features=hidden))

    def forward(self, code_tokens, desc_tokens):

        code_attention_mask = code_tokens['attention_mask']  # (batch_size, code_seq_len)
        code_embeddings = self.codeT5(**code_tokens).last_hidden_state  # (batch_size, code_seq_len, hidden_dim)

        ini_emb = code_embeddings
        desc_embeddings = []
        desc_attention_masks = []

        for des in desc_tokens:
            desc_embeddings.append(self.codeT5(**des).last_hidden_state)
            desc_attention_masks.append(des['attention_mask'])

        combined = list(zip(desc_embeddings, desc_attention_masks))
        random.shuffle(combined)
        desc_embeddings, desc_attention_masks = zip(*combined)

        for layer_idx in range(len(desc_embeddings)):

            desc_attention_mask = desc_attention_masks[layer_idx]
            attention_mask = torch.bmm(
                code_attention_mask.unsqueeze(2).float(),
                desc_attention_mask.unsqueeze(1).float()
            )

            code_embeddings = self.sublayer_connection1[layer_idx](
                code_embeddings,
                lambda _code_embeddings: self.self_attention[layer_idx](
                    _code_embeddings, desc_embeddings[layer_idx], desc_embeddings[layer_idx], attention_mask
                )
            )
            code_embeddings = self.linear_layers[layer_idx](code_embeddings)


        mask = code_attention_mask.unsqueeze(-1).expand(code_embeddings.size()).float()
        masked_embeddings = code_embeddings * mask
        sum_embeddings = masked_embeddings.sum(dim=1)
        valid_token_counts = mask.sum(dim=1)
        mean_embeddings = sum_embeddings / valid_token_counts


        masked_ini_emb = ini_emb * mask
        sum_ini_emb = masked_ini_emb.sum(dim=1)
        mean_ini_emb = sum_ini_emb / valid_token_counts

        return self.fc(mean_embeddings + mean_ini_emb)
