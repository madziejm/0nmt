import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Attention(nn.Module):
    """
    Adapted from https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
    """

    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(
            self.attn(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2))
        )

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)
