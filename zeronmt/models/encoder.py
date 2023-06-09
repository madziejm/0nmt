from typing import Tuple

import torch
from torch import Tensor, nn

from zeronmt.models.datatypes import DimensionSpec, Embeddings, Language


class Encoder(nn.Module):
    """
    Partially adapted from https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
    """

    def __init__(
        self,
        embedding_layer: Embeddings,  # embedding for input tokens
        emb_dim: int,
        dimensions: DimensionSpec,
        dropout: float,
        PAD_IDX: int,
    ):
        super().__init__()

        self.embedding_layer = embedding_layer

        self.special_toks_embedding = nn.Embedding(
            dimensions.nspecial_toks, emb_dim, padding_idx=PAD_IDX
        )

        self.rnn = nn.GRU(emb_dim, dimensions.enc_hid, bidirectional=True)
        self.fc = nn.Linear(dimensions.enc_hid * 2, dimensions.dec_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tok_seq: Tensor, from_lang: Language) -> Tuple[Tensor, Tensor]:
        # hopefully this works
        special_token_mask = tok_seq < self.special_toks_embedding.num_embeddings
        embedded = (
            self.embedding_layer.src(tok_seq)
            if from_lang == Language.src
            else self.embedding_layer.tgt(tok_seq)
        )  # the pretrained embeddings return zero vectors for special tokens
        embedded[special_token_mask] = self.special_toks_embedding(
            tok_seq[special_token_mask]
        )

        embedded = self.dropout(embedded)

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        )

        return outputs, hidden
