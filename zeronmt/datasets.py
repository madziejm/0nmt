import io

from torch.utils.data import Dataset
from torchtext.vocab import Vocab
import torch


class ParallelDataset(Dataset):
    def __init__(
        self,
        src_filepath: str,
        tgt_filepath: str,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        tokenizer,
        min_sentence_len=3,
        max_sentence_len=100,
        max_nsentence=float("inf"),
    ):
        raw_src_iter = iter(io.open(src_filepath, encoding="utf8"))
        raw_tgt_iter = iter(io.open(tgt_filepath, encoding="utf8"))

        self.lens_src = []
        self.lens_tgt = []

        self.src = []
        self.tgt = []
        for i, (raw_de, raw_en) in enumerate(zip(raw_src_iter, raw_tgt_iter)):
            if i >= max_nsentence:
                break
            src_tensor_ = torch.tensor(
                [src_vocab[token] for token in tokenizer(raw_de)], dtype=torch.long
            )
            tgt_tensor_ = torch.tensor(
                [tgt_vocab[token] for token in tokenizer(raw_en)], dtype=torch.long
            )
            self.lens_src.append(len(src_tensor_))
            self.lens_tgt.append(len(tgt_tensor_))
            if (
                min_sentence_len <= len(src_tensor_)
                and len(src_tensor_) <= max_sentence_len
                and min_sentence_len <= len(tgt_tensor_)
                and len(tgt_tensor_) <= max_sentence_len
            ):
                self.src.append(src_tensor_)
                self.tgt.append(tgt_tensor_)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        return self.src[index], self.tgt[index]
