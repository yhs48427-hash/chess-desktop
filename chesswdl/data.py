"""샤드 → 학습용 예시. (Drive chess/chesswdl 의 추론용 사본: 학습용 코드는 뺌)"""
import numpy as np
import torch

from .encoding import elo_bucket, log_clock


def norm_name(name: str) -> str:
    return name.strip().lower()


class GameView:
    def __init__(self, boards, state, clocks, has_clock, evals, has_eval, moves, wdl,
                 legal_flat, legal_offs, white_elo, black_elo, base, inc):
        self.boards, self.state, self.clocks, self.has_clock = boards, state, clocks, has_clock
        self.evals, self.has_eval, self.moves, self.wdl = evals, has_eval, moves, wdl
        self.legal_flat, self.legal_offs = legal_flat, legal_offs
        self.white_elo, self.black_elo, self.base, self.inc = white_elo, black_elo, base, inc
        self.n = len(moves)

    @classmethod
    def from_record(cls, g):
        flat = np.concatenate(g["legal"]).astype(np.int16) if len(g["legal"]) else np.zeros(0, np.int16)
        offs = np.concatenate([[0], np.cumsum([len(x) for x in g["legal"]])]).astype(np.int64)
        return cls(g["boards"], g["state"], g["clocks"], g["has_clock"], g["evals"], g["has_eval"],
                   g["moves"], g["wdl"], flat, offs, int(g["white_elo"]), int(g["black_elo"]),
                   float(g["base"]), float(g["inc"]))


def make_example(gv: GameView, a: int, b: int, t0: int, t1: int, window: int,
                 wid: int = 0, bid: int = 0, user_white=None):
    assert a == 0 or a <= t0 - window + 1
    S = b - a
    ply = np.arange(a, b)
    mover_white = ply % 2 == 0
    we, be = gv.white_elo, gv.black_elo
    my_elo = np.where(mover_white, we, be)
    op_elo = np.where(mover_white, be, we)
    players = np.stack([np.where(mover_white, wid, bid), np.where(mover_white, bid, wid)], 1)
    clocks = np.asarray(gv.clocks[a:b], dtype=np.float32)
    has_clock = np.asarray(gv.has_clock[a:b], dtype=np.float32)
    cont = np.stack([
        log_clock(clocks[:, 0]) * has_clock,
        log_clock(clocks[:, 1]) * has_clock,
        has_clock,
        np.full(S, log_clock(gv.base), dtype=np.float32),
        np.full(S, log_clock(gv.inc * 40), dtype=np.float32),
        np.asarray(gv.evals[a:b], dtype=np.float32),
        np.asarray(gv.has_eval[a:b], dtype=np.float32),
    ], 1)

    t = np.arange(t0, t1)
    t_start = np.maximum(0, t - window + 1) - a
    t_end = t - a
    if user_white is None:
        t_user = np.ones(len(t), dtype=bool)
    else:
        t_user = (t % 2 == 0) == user_white

    out = {
        "boards": torch.from_numpy(np.asarray(gv.boards[a:b], dtype=np.int64)),
        "state": torch.from_numpy(np.asarray(gv.state[a:b], dtype=np.int64)),
        "elo": torch.from_numpy(np.stack([elo_bucket(my_elo), elo_bucket(op_elo)], 1)),
        "cont": torch.from_numpy(cont.astype(np.float32)),
        "players": torch.from_numpy(players.astype(np.int64)),
        "t_start": torch.from_numpy(t_start.astype(np.int64)),
        "t_end": torch.from_numpy(t_end.astype(np.int64)),
        "t_ply": torch.from_numpy(t.astype(np.int64)),
        "t_wdl": torch.from_numpy(np.asarray(gv.wdl[t0:t1], dtype=np.int64)),
        "t_move": torch.from_numpy(np.asarray(gv.moves[t0:t1], dtype=np.int64)),
        "t_user": torch.from_numpy(t_user),
    }
    if gv.legal_offs is not None:
        out["t_legal"] = legal_tensor(gv.legal_flat, gv.legal_offs[t0: t1 + 1])
    return out


def legal_tensor(legal, offs) -> torch.Tensor:
    lens = np.diff(offs)
    K = max(1, int(lens.max()) if len(lens) else 1)
    t = np.full((len(lens), K), -1, dtype=np.int64)
    flat = np.asarray(legal[offs[0]: offs[-1]], dtype=np.int64)
    rows = np.repeat(np.arange(len(lens)), lens)
    cols = np.arange(len(flat)) - np.repeat(offs[:-1] - offs[0], lens)
    t[rows, cols] = flat
    return torch.from_numpy(t)


SEG_KEYS = ("boards", "state", "elo", "cont", "players")
TGT_FILL = {"t_start": 0, "t_end": 0, "t_ply": -1, "t_wdl": -1, "t_move": -1, "t_user": False}


def collate(batch):
    B = len(batch)
    S = max(b["boards"].shape[0] for b in batch)
    M = max(b["t_end"].shape[0] for b in batch)
    out = {}
    for k in SEG_KEYS:
        v = batch[0][k]
        t = torch.zeros((B, S) + tuple(v.shape[1:]), dtype=v.dtype)
        for i, b in enumerate(batch):
            t[i, : b[k].shape[0]] = b[k]
        out[k] = t
    out["mask"] = torch.zeros(B, S, dtype=torch.bool)
    out["t_mask"] = torch.zeros(B, M, dtype=torch.bool)
    for i, b in enumerate(batch):
        out["mask"][i, : b["boards"].shape[0]] = True
        out["t_mask"][i, : b["t_end"].shape[0]] = True
    for k, fill in TGT_FILL.items():
        v = batch[0][k]
        t = torch.full((B, M), fill, dtype=v.dtype)
        for i, b in enumerate(batch):
            t[i, : b[k].shape[0]] = b[k]
        out[k] = t
    out["t_wdl"][out["t_wdl"] == 255] = -1
    if "t_legal" in batch[0]:
        K = max(b["t_legal"].shape[1] for b in batch)
        t = torch.full((B, M, K), -1, dtype=torch.long)
        for i, b in enumerate(batch):
            t[i, : b["t_legal"].shape[0], : b["t_legal"].shape[1]] = b["t_legal"]
        out["t_legal"] = t
    return out
