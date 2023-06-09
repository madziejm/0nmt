from torch import Tensor, nn

from enum import Enum
from typing import Any, NamedTuple
import torchtext.vocab


class Language(str, Enum):
    src = "src"
    tgt = "tgt"


class DimensionSpec(NamedTuple):
    attention: int
    dec_hid: int
    enc_hid: int
    nspecial_toks: int


class Vectors(NamedTuple):
    src: torchtext.vocab.Vectors
    tgt: torchtext.vocab.Vectors


class Embeddings(NamedTuple):
    src: nn.Embedding
    tgt: nn.Embedding
