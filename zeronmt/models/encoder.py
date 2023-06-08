from typing import Tuple

import torch
from torch import Tensor, nn


class Encoder(nn.Module):
    """
    Adapted from https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
    """

    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Tensor]:
        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        )

        return outputs, hidden
