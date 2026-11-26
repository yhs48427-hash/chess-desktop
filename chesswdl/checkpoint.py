import os

import torch

from .model import ChessWDLModel


def atomic_save(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(path, model, vocab, metrics=None, extra=None):
    atomic_save({"model": model.state_dict(), "cfg": model.cfg, "num_players": model.num_players,
                 "vocab": vocab, "metrics": metrics, "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ChessWDLModel(ck["cfg"], ck["num_players"])
    model.load_state_dict(ck["model"])
    return model.to(device), ck["vocab"], ck
