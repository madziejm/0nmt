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
        )  # skip the leading BOS token, leave the number of output class dim, so Pytorch knows we are passing logits here
        l_src_tgt = self.criterion(
            output, tgt[1:].view(-1)
        )  # skip the leading BOS token (that has already been an input), flatten (this tensor contains class id-s, not logits)

        # # tgt -> src step
        output = self(tgt, src, Language.tgt, Language.src, teacher_forcing)
        output = output[1:].view(
            -1, output.shape[-1]
        )  # skip the leading BOS token, leave the number of output class dim, so Pytorch knows we are passing logits here

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
    Return loss based on criteria from 1711.00043
    """

    @staticmethod
    def apply_permutation(sentence: torch.Tensor):
        n = len(sentence)
        k = 3
        alpha = k + 1
        q = torch.arange(n) + torch.rand(n) * alpha
        sorted_indices = sorted(range(n), key=lambda x: q[x])

        # shuffled_sentence = [sentence[i] for i in sorted_indices]
        return sentence[sorted_indices]

    def corrupt_tokens(self, monoling_batch):
        """
        drop and reoder tokens at random
        """
        # # dropping is buggy, skip it
        # # p_wd = 0.1  # probability of word drop; value from arxiv:1711.00043
        EOS_IDX = 3  # pardon for hardcoding it here TODO
        # # allowed_idx = torch.arange(1, len(monoling_batch -1))
        # # allowed_idx = allowed_idx[(monoling_batch[allowed_idx,:] != EOS_IDX).all(dim=1)]
        # # word_idx_to_drop = (
        # #     allowed_idx[
        # #         torch.rand(len(allowed_idx)) <= p_wd
        # #     ]

        # # )  # to recall, 1st dim is #words, 2nd dim is monoling_batch size # we want to preserve BOS and EOS tokens
        # monoling_batch = monoling_batch[~word_idx_to_drop, :]
        for i in range(monoling_batch.shape[1]):
            sent = monoling_batch[:, i]
            sent_end = sent[sent == EOS_IDX][0]  # wow wow so efficient
            sent[1:sent_end] = self.apply_permutation(sent[1:sent_end])
            monoling_batch[:, i] = sent
        return monoling_batch

    def cross_domain_step(self, src, tgt, teacher_forcing: float = 0.0):
        # src, tgt === source and target batch

        # src -> tgt -> src step
        output = self(src, tgt, Language.src, Language.tgt, teacher_forcing)
        output = self(
            output.argmax(-1),
            src,
            Language.tgt,
            Language.src,
            teacher_forcing,  # argmax as we want tokens, not logits
        )

        l_cd_src = self.criterion(
            output[1:].view(
                -1, output.shape[-1]
            ),  # skip the leading BOS token, leave the number of output class dim, so Pytorch knows we are passing logits here
            src[1:].view(-1),
        )  # skip the leading BOS token (that has already been an input), flatten (this tensor contains class id-s, not logits)

        # tgt -> src -> tgt step
        output = self(tgt, src, Language.tgt, Language.src, teacher_forcing)
        output = self(
            output.argmax(-1),
            tgt,
            Language.src,
            Language.tgt,
            teacher_forcing,  # argmax as we want tokens, not logits
        )

        l_cd_tgt = self.criterion(
            output[1:].view(
                -1, output.shape[-1]
            ),  # skip the leading BOS token, leave the number of output class dim, so Pytorch knows we are passing logits here
            tgt[1:].view(-1),
        )  # skip the leading BOS token (that has already been an input), flatten (this tensor contains class id-s, not logits)

        return l_cd_src, l_cd_tgt

    def base_step(self, batch, teacher_forcing: float = 0.0, mode: str = "train"):
        batch_size = len(batch)
        src, tgt = batch

        # --- Denoising Auto - Encoding ---
        # TODO apply copy the original output
        # apply noise to the input before passing it to the model
        # calculate loss based on the input before applying noise
        output_src = self(
            self.corrupt_tokens(src.clone()),
            src,
            Language.src,
            Language.src,
            teacher_forcing,
        )
        output_tgt = self(
            self.corrupt_tokens(tgt.clone()),
            tgt,
            Language.tgt,
            Language.tgt,
            teacher_forcing,
        )
        l_auto = self.criterion(
            output_src[1:].view(-1, output_src.shape[-1]),
            src[1:].view(
                -1
            ),  # skip the leading BOS token (that has already been an input), flatten (this tensor contains class id-s, not logits)
        ) + self.criterion(
            output_tgt[1:].view(-1, output_tgt.shape[-1]),
            tgt[1:].view(
                -1
            ),  # skip the leading BOS token (that has already been an input), flatten (this tensor contains class id-s, not logits)
        )

        l_cd_src, l_cd_tgt = self.cross_domain_step(src, tgt, teacher_forcing)
        l_cd = l_cd_src + l_cd_tgt

        # --- Adversarial Training --- # TODO
        l_adv = 0.0

        loss = l_auto + l_cd + l_adv

        self.log(mode + "_loss_auto", l_auto, prog_bar=True, batch_size=batch_size)
        self.log(mode + "_loss_cd_src", l_cd_src, prog_bar=True, batch_size=batch_size)
        self.log(mode + "_loss_cd_tgt", l_cd_tgt, prog_bar=True, batch_size=batch_size)
        self.log(mode + "_loss_cd", l_cd, prog_bar=True, batch_size=batch_size)
        self.log(mode + "_loss_adv", l_adv, prog_bar=True, batch_size=batch_size)
        self.log(mode + "_loss", loss, prog_bar=True, batch_size=batch_size)

        # gc.collect() # TODO consider using me when really in need of memory
        # torch.cuda.empty_cache() # TODO consider using me when really in need of memory

        return loss
