from typing import Tuple

import torch
from torch import Tensor, nn

from zeronmt.models.attention import Attention
from zeronmt.models.datatypes import DimensionSpec, Embeddings, Language


class Decoder(nn.Module):
    """
    Adapted from https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
    """

    def __init__(
        self,
        embedding_layer: Embeddings,  # embedding for output tokens
        emb_dim: int,
        dimensions: DimensionSpec,
        dropout: float,
        attention: Attention,
        PAD_IDX: int,
    ):
        super().__init__()

        self.embedding_layer = embedding_layer
        for k, emb in self.embedding_layer._asdict().items():
            self.add_module("embedding_" + k, emb)

        self.attention = attention

        self.special_toks_embedding = nn.Embedding(
            dimensions.nspecial_toks, emb_dim, padding_idx=PAD_IDX
        )

        self.rnn = nn.GRU((dimensions.enc_hid * 2) + emb_dim, dimensions.dec_hid)

        self.output_to_src = nn.Linear(
            self.attention.attn_in + emb_dim, self.embedding_layer.src.num_embeddings
        )
        self.output_to_tgt = nn.Linear(
            self.attention.attn_in + emb_dim, self.embedding_layer.tgt.num_embeddings
        )

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
        self,
        input_tok_seq: Tensor,
        decoder_hidden: Tensor,
        encoder_outputs: Tensor,
        to_lang: Language,
    ) -> Tuple[Tensor, Tensor]:
        input_tok_seq = input_tok_seq.unsqueeze(0)

        # hopefully this works
        special_token_mask = input_tok_seq < self.special_toks_embedding.num_embeddings
        embedded = (
            self.embedding_layer.tgt(  # these are zeros for special tokens
                input_tok_seq
            )
            if to_lang == Language.tgt
            else self.embedding_layer.src(
                input_tok_seq
            )  # these are zeros for special tokens
        )
        embedded[
            special_token_mask
        ] = self.special_toks_embedding(  # nonzero only for special tokens
            input_tok_seq[special_token_mask]
        )

        embedded = self.dropout(embedded)

        weighted_encoder_rep = self._weighted_encoder_rep(
            decoder_hidden, encoder_outputs
        )

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output: Tensor = (
            self.output_to_src(
                torch.cat((output, weighted_encoder_rep, embedded), dim=1)
            )
            if to_lang == Language.src
            else self.output_to_tgt(
                torch.cat((output, weighted_encoder_rep, embedded), dim=1)
            )
        ).to(input_tok_seq.device)

        return output, decoder_hidden.squeeze(0)
