import torch
import torch.nn as nn
from Multihead_Attention import DecoderAttention
from SubLayerConnection import SublayerConnection
from transformers import T5EncoderModel
import random

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
            self.linear_layers.append(nn.Linear(in_features=args.hidden_dim, out_features=args.hidden_dim))

    def forward(self, code_tokens, desc_tokens):
        code_embeddings = self.codeT5(**code_tokens).last_hidden_state
        desc_tokens = [self.codeT5(**des).last_hidden_state for des in desc_tokens]
        random.shuffle(desc_tokens)

        for layer_idx in range(self.num_layers):
            resnet = code_embeddings
            code_embeddings = self.sublayer_connection1[layer_idx](code_embeddings,
                                                              lambda _code_embeddings: self.self_attention[
                                                                  layer_idx].forward(_code_embeddings, desc_tokens[layer_idx],
                                                                                     desc_tokens[layer_idx]))
            code_embeddings = self.linear_layers[layer_idx](code_embeddings) + resnet
        return self.fc(code_embeddings.mean(dim=1))