# from zeronmt.models.discriminator import Discriminator # TODO
# import gc
import random
from typing import Any

import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn

from zeronmt.models.attention import Attention
from zeronmt.models.datatypes import DimensionSpec, Embeddings, Language, Vectors
from zeronmt.models.decoder import Decoder
from zeronmt.models.encoder import Encoder


class Seq2Seq(pl.LightningModule):
    def __init__(
        self,
        dec_dropout: float,
        enc_dropout: float,
        dimensions: DimensionSpec,
        PAD_IDX: int,
        pretrained_embeddings: Vectors,
        teacher_forcing_ratio=0.5,
    ):
        super().__init__()

        emb_dim: int = pretrained_embeddings.src.dim
        assert emb_dim is not None, "word embedding length is not initialized, set it"
        assert (
            pretrained_embeddings.src.dim == pretrained_embeddings.tgt.dim
        ), "we wanted the embedding to interchangable, right?"

        # TODO consider normalization here
        # we'll share this layer just in case to prevent tensor copies
        self.embedding_layer = Embeddings(
            nn.Embedding.from_pretrained(
                pretrained_embeddings.src.vectors,
                freeze=True,
            ),
            nn.Embedding.from_pretrained(
                pretrained_embeddings.tgt.vectors,
                freeze=True,
            ),
        )
        for k, emb in self.embedding_layer._asdict().items():
            self.add_module("embedding_" + k, emb)

        self.encoder = Encoder(
            self.embedding_layer,
            emb_dim,
            dimensions,
            enc_dropout,
            PAD_IDX,
        )
        attn = Attention(dimensions.enc_hid, dimensions.dec_hid, dimensions.attention)
        self.decoder = Decoder(
            self.embedding_layer,
            emb_dim,
            dimensions,
            dec_dropout,
            attn,
            PAD_IDX,
        )

        self.criterion = (
            nn.CrossEntropyLoss()
        )  # ignore_index=PAD_IDX) # not needed TODO remove (ignoring is handled by nn.Embedding)
        self.teacher_forcing_ratio = teacher_forcing_ratio

        for name, param in self.named_parameters():
            print(name)
        # TODO remove this intialization
        #     if "weight" in name:
        #         nn.init.normal_(param.data, mean=0, std=0.01)
        #     else:
        #         nn.init.constant_(param.data, 0)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        from_lang: Language,
        to_lang: Language,
        teacher_forcing_ratio,
    ) -> Tensor:
        batch_size = src.shape[1]

        # lenghts of source and target need to be equal in order to prevent
        # Expected input batch_size (xxx) to match target batch_size (xxx).
        # when calculating the loss
        max_output_len = tgt.shape[0]
        tgt_vocab_size = self.embedding_layer.tgt.num_embeddings

        outputs = torch.zeros(max_output_len, batch_size, tgt_vocab_size).to(src.device)

        encoder_outputs, hidden = self.encoder(src, from_lang)

        # first input to the decoder is the <bos> token, copy it from the target sequence
        output = tgt[0, :]

        for t in range(1, max_output_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs, to_lang)
            outputs[t] = output
            teacher_force = (
                False
                if teacher_forcing_ratio == 0.0
                else (random.random() < teacher_forcing_ratio)
            )
            top_token = output.argmax(-1)
            output = tgt[t] if teacher_force else top_token

        return outputs

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters())  # TODO customize lr?

    def base_step(self, batch, teacher_forcing: float = 0.0, mode: str = "train"):
        batch_size = len(batch)
        src, tgt = batch

        # --- Denoising Auto - Encoding ---

        # src -> tgt step
        output = self(src, tgt, Language.src, Language.tgt, teacher_forcing)
        output = output[1:].view(
            -1, output.shape[-1]
        )  # skip the leading BOS token, leave the number of output class dim, do Pytorch knows we are passing logits here
        l_autoenc_src_tgt = self.criterion(
            output, tgt[1:].view(-1)
        )  # skip the leading BOS token (that has already been an input), flatten (this tensor contains class id-s, not logits)

        # # tgt -> src step
        output = self(tgt, src, Language.tgt, Language.src, teacher_forcing)
        output = output[1:].view(
            -1, output.shape[-1]
        )  # skip the leading BOS token, leave the number of output class dim, do Pytorch knows we are passing logits here

        l_autoenc_tgt_src = self.criterion(
            output, src[1:].view(-1)
        )  # skip the leading BOS token (that has already been an input), flatten (this tensor contains class id-s, not logits)

        # --- Cross Domain Training ---
        l_cd = 0.0
        # --- Adversarial Training ---
        l_adv = 0.0

        loss = l_autoenc_src_tgt + l_autoenc_tgt_src + l_cd + l_adv

        self.log(mode + "_loss", loss, prog_bar=True, batch_size=batch_size)

        # gc.collect() # TODO consider using me when really in need of memory
        # torch.cuda.empty_cache() # TODO consider using me when really in need of memory

        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        assert 1 == len(batch)
        return self.base_step(batch[0], self.teacher_forcing_ratio)

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.base_step(batch, teacher_forcing=0.0, mode="val")

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.base_step(batch, teacher_forcing=0.0, mode="test")
