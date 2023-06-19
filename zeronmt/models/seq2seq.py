# from zeronmt.models.discriminator import Discriminator # TODO
# import gc
import random
from typing import Any

import pytorch_lightning as pl
import torch
import torch.optim as optim
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn
import torch.nn.functional as F

from zeronmt.models.attention import Attention
from zeronmt.models.datatypes import DimensionSpec, Embeddings, Language, Vectors
from zeronmt.models.decoder import Decoder
from zeronmt.models.encoder import Encoder


class Discriminator(pl.LightningModule):
    def __init__(self, input_size, embedding_size, hidden_size=1024):
        super(self).__init__()

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.hidden_layer1 = nn.Linear(embedding_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer3 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = self.hidden_layer1(x)
        x = self.leaky_relu(x)
        x = self.hidden_layer2(x)
        x = self.leaky_relu(x)
        x = self.hidden_layer3(x)
        x = self.leaky_relu(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

    def training_step(self, batch, labels):
    
        predictions = self(batch)
        loss = F.binary_cross_entropy(predictions, labels)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


class Seq2SeqSupervised(pl.LightningModule):
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
        """
        Get a batch of really parallel pairs of src-tgt sentences.
        Translate src -> tgt and tgt -> src.
        Return loss based the the two-way translations.
        This is similar to CROSS DOMAIN TRAINING from arXiv:1711.00043.
        """
        batch_size = len(batch)
        src, tgt = batch

        # src -> tgt step
        output = self(src, tgt, Language.src, Language.tgt, teacher_forcing)
        output = output[1:].view(
            -1, output.shape[-1]
        )  # skip the leading BOS token, leave the number of output class dim, do Pytorch knows we are passing logits here
        l_src_tgt = self.criterion(
            output, tgt[1:].view(-1)
        )  # skip the leading BOS token (that has already been an input), flatten (this tensor contains class id-s, not logits)

        # # tgt -> src step
        output = self(tgt, src, Language.tgt, Language.src, teacher_forcing)
        output = output[1:].view(
            -1, output.shape[-1]
        )  # skip the leading BOS token, leave the number of output class dim, do Pytorch knows we are passing logits here

        l_tgt_src = self.criterion(
            output, src[1:].view(-1)
        )  # skip the leading BOS token (that has already been an input), flatten (this tensor contains class id-s, not logits)

        loss = l_src_tgt + l_tgt_src

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


class Seq2SeqUnsupervised(Seq2SeqSupervised):
    """
    Get a batch of pairs of src-tgt sentences that are not aligned.
    Translate src -> tgt and tgt -> src.
    Return loss based the the two-way translations.
    """
    def __init__(
        self,
        dec_dropout: float,
        enc_dropout: float,
        dimensions: DimensionSpec,
        PAD_IDX: int,
        batch_size:int,
        pretrained_embeddings: Vectors,
        teacher_forcing_ratio=0.5,
    ):
        super().__init__()

        emb_dim: int = pretrained_embeddings.src.dim
        assert emb_dim is not None, "word embedding length is not initialized, set it"
        assert (
            pretrained_embeddings.src.dim == pretrained_embeddings.tgt.dim
        ), "we wanted the embedding to interchangable, right?"

        self.discriminator = Discriminator(batch_size, emb_dim)

    def base_step(self, batch, teacher_forcing: float = 0.0, mode: str = "train"):
        batch_size = len(batch)
        src, tgt = batch

        # --- Denoising Auto - Encoding ---
        # TODO
        l_auto = 0.0

        # --- Cross Domain Training ---
        # TODO make this step resemble the corresponding step from arXiv:1711.00043
        # src -> tgt step
        output = self(src, tgt, Language.src, Language.tgt, teacher_forcing)
        output = output[1:].view(
            -1, output.shape[-1]
        )  # skip the leading BOS token, leave the number of output class dim, do Pytorch knows we are passing logits here
        l_cd_src_tgt = self.criterion(
            output, tgt[1:].view(-1)
        )  # skip the leading BOS token (that has already been an input), flatten (this tensor contains class id-s, not logits)

        # # tgt -> src step
        output = self(tgt, src, Language.tgt, Language.src, teacher_forcing)
        output = output[1:].view(
            -1, output.shape[-1]
        )  # skip the leading BOS token, leave the number of output class dim, do Pytorch knows we are passing logits here

        l_cd_tgt_src = self.criterion(
            output, src[1:].view(-1)
        )  # skip the leading BOS token (that has already been an input), flatten (this tensor contains class id-s, not logits)

        # --- Cross Domain Training --- # TODO
        l_cd = 0.0
        # --- Adversarial Training --- # TODO

        dis_batch, dis_labels = "todo", "todo" ### need to shuffle tgt and src in batch, labels according to that split
        discriminator_loss = self.discriminator.training_step(self, dis_batch, dis_labels)
        l_adv = 1.0 - discriminator_loss #?????

        loss = l_auto + l_cd_src_tgt + l_cd_tgt_src + l_adv

        self.log(mode + "_loss", loss, prog_bar=True, batch_size=batch_size)

        # gc.collect() # TODO consider using me when really in need of memory
        # torch.cuda.empty_cache() # TODO consider using me when really in need of memory

        return loss
