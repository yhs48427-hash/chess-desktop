"""수별 가중 승률 (move-weighted WDL). (Drive chess/chesswdl 의 추론용 사본)"""
import json

import chess
import numpy as np
import torch

from .encoding import board_tokens, index_to_move, log_clock, state_features
from .policy import DEFAULT_TOP_P, select_candidates

WIN, DRAW, LOSS = 0, 1, 2
CALIB_FEATURES = ["eval", "elo_diff", "clock_ratio", "has_clock", "progress"]


def terminal_wdl(board: chess.Board):
    if board.is_checkmate():
        return np.array([0.0, 0.0, 1.0])
    if board.is_stalemate() or board.is_insufficient_material():
        return np.array([0.0, 1.0, 0.0])
    return None


def flip(wdl):
    return np.array([wdl[LOSS], wdl[DRAW], wdl[WIN]])


def flip_range(rng):
    return np.array([rng[LOSS], rng[DRAW], rng[WIN]])


def flip_note(txt):
    if txt is None:
        return None
    if txt.startswith("#"):
        m = txt[1:]
        return "#" + (m[1:] if m.startswith("-") else "-" + m)
    return f"{-float(txt):+.2f}"


class EngineValue:
    def __init__(self, engine, calib_path):
        self.engine = engine
        with open(calib_path, encoding="utf-8") as f:
            c = json.load(f)
        assert c["features"] == CALIB_FEATURES
        self.W = np.asarray(c["W"], dtype=np.float64)
        self.b = np.asarray(c["b"], dtype=np.float64)

    def wdl(self, board, mover_elo, opp_elo, mover_clock, opp_clock, has_clock, ply):
        ev, txt = self.engine.evaluate(board)
        x = np.array([ev, (mover_elo - opp_elo) / 400.0,
                      (float(log_clock(mover_clock)) - float(log_clock(opp_clock))) * has_clock,
                      has_clock, min(ply / 100.0, 2.0)])
        z = x @ self.W + self.b
        z = np.exp(z - z.max())
        return z / z.sum(), txt


@torch.no_grad()
def move_weighted_wdl(model, batch, h, t, board, pol_t, top_p=DEFAULT_TOP_P, value="model",
                      engine_value=None, raw=None, engine=None):
    idxs, probs, cum = select_candidates(pol_t, top_p)
    moves = [index_to_move(board, i) for i in idxs]
    if any(m is None for m in moves):
        raise RuntimeError("정책 후보에 합법수가 아닌 수가 있습니다 (합법수 마스크 확인 필요).")

    results = [None] * len(moves)
    notes = [None] * len(moves)
    children = []
    for i, m in enumerate(moves):
        b2 = board.copy(stack=False)
        b2.push(m)
        tw = terminal_wdl(b2)
        if tw is not None:
            results[i] = flip(tw)
        else:
            children.append((i, b2))

    if children and value == "model":
        W = model.cfg["window"]
        dev = h.device
        n = len(children)
        st = max(0, t - W + 2)
        elo, cont, pl = batch["elo"][0, t], batch["cont"][0, t], batch["players"][0, t]
        c_cont = cont.clone()
        c_cont[0], c_cont[1] = cont[1], cont[0]
        c_cont = c_cont.unsqueeze(0).repeat(n, 1)
        c_cont[:, 5] = 0.0
        c_cont[:, 6] = 0.0
        if engine is not None:
            for j, (i, b2) in enumerate(children):
                ev, txt = engine.evaluate(b2)
                c_cont[j, 5], c_cont[j, 6] = ev, 1.0
                notes[i] = txt
        hb = torch.from_numpy(np.stack([board_tokens(b) for _, b in children]).astype(np.int64)).to(dev)
        sb = torch.from_numpy(np.stack([state_features(b, t + 1) for _, b in children]).astype(np.int64)).to(dev)
        hc = model.spatial(hb, sb, elo.flip(0).expand(n, 2), c_cont)
        seq = torch.cat([h[st: t + 1].unsqueeze(0).expand(n, -1, -1), hc.unsqueeze(1)], 1)
        pseq = torch.cat([batch["players"][0, st: t + 1].unsqueeze(0).expand(n, -1, -1),
                          pl.flip(0).view(1, 1, 2).expand(n, 1, 2)], 1)
        lens = torch.full((n,), seq.shape[1], dtype=torch.long, device=dev)
        cw = model.wdl_head(model.last_features(seq, pseq, lens)).float().softmax(-1).cpu().numpy()
        for (i, _), w in zip(children, cw):
            results[i] = flip(w)
    elif children and value == "engine":
        r = raw
        for i, b2 in children:
            w, txt = engine_value.wdl(b2, r["opp_elo"], r["mover_elo"], r["opp_clock"], r["mover_clock"],
                                      r["has_clock"], t + 1)
            results[i] = flip(w)
            notes[i] = txt
    elif children:
        raise ValueError(f"알 수 없는 평가 방식: {value}")

    P = np.asarray(probs)
    R = np.stack(results)
    C = float(P.sum())
    partial = (P[:, None] * R).sum(0)
    conditional = partial / C
    excluded = max(0.0, 1.0 - C)
    rng = np.stack([partial, np.minimum(1.0, partial + excluded)], 1)
    running = np.cumsum(P)
    cands = [{"san": board.san(m), "uci": m.uci(), "idx": int(ix), "p": float(p), "cum": float(c),
              "wdl": res, "engine": None if nt is None else flip_note(nt)}
             for m, ix, p, c, res, nt in zip(moves, idxs, P, running, R, notes)]
    return {"n_legal": board.legal_moves.count(), "top_p": top_p, "candidates": cands,
            "selected_mass": C, "excluded_mass": excluded, "cumulative": cum,
            "conditional": conditional, "partial": partial, "range": rng}
