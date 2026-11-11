"""Stay-level dataset: one item = one admission (its full causal token sequence and all of its landmarks)."""
from __future__ import annotations

from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

TOKEN_FIELDS = ("concept", "ingredient", "modality", "route", "phase", "dose_level", "value", "value_mask",
                "time", "na_load", "node")
LONG_FIELDS = {"concept", "ingredient", "modality", "route", "phase", "dose_level", "node"}


def load_pack(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        pack = {k: z[k] for k in z.files}
    validate_pack(pack)
    return pack


def validate_pack(pack: dict) -> None:
    """Validate the stay boundaries and landmark positions before batching."""
    def array(name):
        if name not in pack:
            raise ValueError(f"pack is missing {name}")
        return np.asarray(pack[name])

    stays = array("stay_id")
    if stays.ndim != 1:
        raise ValueError("stay_id must be one-dimensional")

    def offsets(name, total):
        values = array(name)
        if (values.ndim != 1 or values.size != stays.size + 1
                or values.dtype.kind not in "iu" or values[0] != 0
                or values[-1] != total or (values[1:] < values[:-1]).any()):
            raise ValueError(f"{name} must partition all rows in stay order")
        return values.astype(np.int64)

    token_time = array("tok_time")
    if token_time.ndim != 1:
        raise ValueError("tok_time must be one-dimensional")
    tok = offsets("tok_offsets", token_time.size)
    for field in TOKEN_FIELDS:
        name = f"tok_{field}"
        if array(name).shape != token_time.shape:
            raise ValueError(f"{name} must align with tok_time")
    positions = array("lmk_pos")
    if positions.ndim != 1 or positions.dtype.kind not in "iu":
        raise ValueError("lmk_pos must be a one-dimensional integer vector")
    lmk = offsets("lmk_offsets", positions.size)
    lengths = np.repeat(np.diff(tok), np.diff(lmk))
    if ((positions < 0) | (positions >= lengths)).any():
        raise ValueError("lmk_pos must index a token within its stay")
    for name in ("lmk_time", "lmk_mode", "lmk_na_last", "lmk_na_mask", "node_feat",
                 *(key for key in pack if key.startswith("y_"))):
        values = array(name)
        if values.ndim < 1 or values.shape[0] != positions.size:
            raise ValueError(f"{name} must align with lmk_pos")



def target_keys(pack: dict) -> list[str]:
    return sorted(k[2:] for k in pack if k.startswith("y_"))


class StayDataset(Dataset):
    def __init__(self, pack: dict):
        self.p = pack
        self.lengths = np.diff(pack["tok_offsets"])
        self.targets = target_keys(pack)

    def __len__(self) -> int:
        return len(self.p["stay_id"])

    def __getitem__(self, i: int) -> int:
        return i


class Collator:
    """Right-pads token sequences; landmarks of all stays in the batch are flattened with their batch index."""

    def __init__(self, pack: dict, n_nodes_pad: int):
        self.p = pack
        self.n_nodes_pad = n_nodes_pad
        self.targets = target_keys(pack)

    def __call__(self, idx: list[int]) -> dict:
        p = self.p
        starts, ends = p["tok_offsets"][idx], p["tok_offsets"][np.asarray(idx) + 1]
        L = int((ends - starts).max())
        B = len(idx)
        out = {}
        for f in TOKEN_FIELDS:
            src = p[f"tok_{f}"]
            pad_value = self.n_nodes_pad if f == "node" else 0
            arr = np.full((B, L), pad_value, dtype=np.int64 if f in LONG_FIELDS else np.float32)
            for b, (s, e) in enumerate(zip(starts, ends)):
                arr[b, : e - s] = src[s:e]
            out[f] = torch.from_numpy(arr)
        mask = np.zeros((B, L), dtype=bool)
        for b, (s, e) in enumerate(zip(starts, ends)):
            mask[b, : e - s] = True
        out["pad_mask"] = torch.from_numpy(mask)

        lmk_rows, lmk_batch = [], []
        for b, i in enumerate(idx):
            a, e = p["lmk_offsets"][i], p["lmk_offsets"][i + 1]
            lmk_rows.append(np.arange(a, e))
            lmk_batch.append(np.full(e - a, b))
        rows = np.concatenate(lmk_rows)
        out["lmk_index"] = torch.from_numpy(rows.astype(np.int64))
        out["lmk_batch"] = torch.from_numpy(np.concatenate(lmk_batch).astype(np.int64))
        out["lmk_pos"] = torch.from_numpy(p["lmk_pos"][rows].astype(np.int64))
        out["lmk_time"] = torch.from_numpy(p["lmk_time"][rows])
        out["lmk_mode"] = torch.from_numpy(p["lmk_mode"][rows].astype(np.int64))
        out["lmk_na_last"] = torch.from_numpy(p["lmk_na_last"][rows])
        out["lmk_na_mask"] = torch.from_numpy(p["lmk_na_mask"][rows].astype(np.float32))
        out["node_feat"] = torch.from_numpy(p["node_feat"][rows])
        out["static"] = torch.from_numpy(p["static"][idx])
        out["targets"] = {k: torch.from_numpy(p[f"y_{k}"][rows]) for k in self.targets}
        return out


class LengthBucketSampler(Sampler):
    """Batches of similar sequence length (less padding), reshuffled every epoch."""

    def __init__(self, lengths: np.ndarray, batch_size: int, shuffle: bool = True, seed: int = 0, mult: int = 50):
        for name, size in (("batch_size", batch_size), ("mult", mult)):
            if isinstance(size, bool) or not isinstance(size, Integral) or size <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.lengths, self.bs, self.shuffle, self.seed, self.mult, self.epoch = lengths, int(batch_size), shuffle, seed, int(mult), 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))
        batches = []
        for s in range(0, len(order), self.bs * self.mult):
            chunk = order[s:s + self.bs * self.mult]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            batches += [chunk[j:j + self.bs].tolist() for j in range(0, len(chunk), self.bs)]
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return int(np.ceil(len(self.lengths) / self.bs))
