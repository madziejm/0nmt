import os.path
from typing import List, Optional

import numpy.typing as npt
import torch
from torchtext.vocab import Vectors


class FastTextAligned(Vectors):
    """
    Downlaods and optionally maps FastText word embeddings.
    Use "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.{}.align.vec" % lang for already aligned word embeddings or
    use non-aligned FastTExt embeddings from url="https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{}.vec" % for lang with custom mapping that aligns the embeddgins.
    """

    def __init__(
        self,
        url: str,
        # filename: str,
        special_toks: List[str],
        mapping: Optional[npt.NDArray] = None,
        **kwargs
    ) -> None:
        name = os.path.basename(url)
        super().__init__(name, url=url, **kwargs)

        if mapping is not None:
            self.vectors @= mapping

        # prepend special_tokens tokens
        self.itos[0:0] = special_toks

        # hopefully it is not slow :)
        self.stoi = {
            **dict(zip(special_toks, range(len(special_toks)))),
            **{word: i + len(special_toks) for i, word in enumerate(self.stoi)},
        }

        # the vectors for the special tokens here will not be used by the model
        # we set them to zeros so indexing works flawlessly
        vecs_special_toks = torch.zeros(len(special_toks), self.dim)
        self.vectors = torch.cat((vecs_special_toks, self.vectors), dim=0)
        assert len(self.vectors) == len(self.itos)
        assert len(self.vectors) == len(self.stoi)
