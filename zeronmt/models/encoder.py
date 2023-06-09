from typing import Tuple

import torch
from torch import Tensor, nn

# from zeronmt.models.attention import MultiheadAttention
from torchtext.vocab import Vectors


class Encoder(nn.Module):
    """
    Partially adapted from https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
    """

    def __init__(
        self,
        input_dim: int,
        # emb_dim: int,
        pretrained_embeddings: Vectors,  # input embeddings
        enc_hid_dim: int,
        dec_hid_dim: int,
        dropout: float,
        PAD_IDX: int,
        num_special_toks: int,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = pretrained_embeddings.dim
        assert (
            self.emb_dim is not None
        ), "word embedding length is not initialized, set it"
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        self.num_special_toks = num_special_toks

        self.special_toks_embedding = nn.Embedding(
            self.num_special_toks, self.emb_dim, padding_idx=PAD_IDX
        )
        # TODO consider normalization here
        self.pretrained_embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings.vectors,
            freeze=True,
        )

        self.rnn = nn.GRU(self.emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Tensor]:
        # hopefully this works
        special_token_mask = src < self.special_toks_embedding.num_embeddings
        embedded = self.pretrained_embedding(src)  # these are zeros for special tokens
        embedded[special_token_mask] = self.special_toks_embedding(
            src[special_token_mask]
        )

        embedded = self.dropout(embedded)

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        )

        return outputs, hidden
