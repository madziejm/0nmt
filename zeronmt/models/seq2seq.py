import random
from typing import Any

import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn

from zeronmt.models.decoder import Decoder
from zeronmt.models.encoder import Encoder


class Seq2Seq(pl.LightningModule):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        PAD_IDX: int,
        teacher_forcing_ratio=0.5,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.save_hyperparameters()

        # TODO remove this intialization?
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    def forward(self, src: Tensor, tgt: Tensor, teacher_forcing_ratio) -> Tensor:
        batch_size = src.shape[1]

        # lenghts of source and target need to be equal in order to prevent
        # Expected input batch_size (xxx) to match target batch_size (xxx).
        # when calculatint the loss
        max_output_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_output_len, batch_size, tgt_vocab_size)

        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <bos> token
        output = tgt[0, :]

        for t in range(1, max_output_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = (
                False
                if teacher_forcing_ratio == 0.0
                else (random.random() < teacher_forcing_ratio)
            )  # to tak dziala? TODO
            top_token = output.argmax(-1)
            output = tgt[t] if teacher_force else top_token

        return outputs

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.parameters())  # TODO cusotmize lr?

    def base_step(self, batch, teacher_forcing: float = 0.0, mode: str = "train"):
        src, tgt = batch
        output = self(src, tgt, teacher_forcing)

        output = output[1:].view(-1, output.shape[-1])
        tgt = tgt[1:].view(-1)

        loss = self.criterion(output, tgt)

        self.log(mode + "_loss", loss, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        assert 1 == len(batch)
        return self.base_step(batch[0], self.teacher_forcing_ratio)

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.base_step(batch, teacher_forcing=0.0)

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        print(f"test: {batch_idx}")
        return self.base_step(batch, teacher_forcing=0.0)
