from typing import Tuple

import torch
from torch import Tensor, nn

from zeronmt.models.attention import Attention


class Decoder(nn.Module):
    """
    Adapted from https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
    """

    def __init__(
        self,
        output_dim: int,
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        dropout: int,
        attention: Attention,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(
        self, decoder_hidden: Tensor, encoder_outputs: Tensor
    ) -> Tensor:
        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep

    def forward(
        self, input: Tensor, decoder_hidden: Tensor, encoder_outputs: Tensor
    ) -> Tuple[Tensor]:
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        weighted_encoder_rep = self._weighted_encoder_rep(
            decoder_hidden, encoder_outputs
        )

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output, weighted_encoder_rep, embedded), dim=1))

        return output, decoder_hidden.squeeze(0)
