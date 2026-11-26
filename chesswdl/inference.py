"""추론 공통 함수. (Drive chess/chesswdl 의 추론용 사본)"""
import chess
import numpy as np
import torch

from .data import GameView, collate, make_example
from .encoding import index_to_move
from .policy import mask_illegal


def game_batch(gv: GameView, window: int, wid=0, bid=0, user_white=None):
    return collate([make_example(gv, 0, gv.n, 0, gv.n, window, wid, bid, user_white)])


@torch.no_grad()
def run_game(model, batch, device):
    model.eval()
    batch = {k: v.to(device) for k, v in batch.items()}
    h = model.encode_positions(batch)
    wdl_logits, pol_logits = model.heads_at(h, batch["players"], batch["t_start"], batch["t_end"], batch["t_mask"])
    wdl = wdl_logits[0].float().softmax(-1).cpu().numpy()
    pol = mask_illegal(pol_logits, batch.get("t_legal"))[0].softmax(-1).cpu()
    return batch, h[0], wdl, pol


def replay_boards(move_indices):
    b = chess.Board()
    boards = []
    for mi in move_indices:
        boards.append(b.copy(stack=False))
        mi = int(mi)
        if mi < 0:
            break
        m = index_to_move(b, mi)
        if m is None:
            raise ValueError(f"ply {len(boards) - 1}: 정책 번호 {mi}가 합법수가 아닙니다.")
        b.push(m)
    return boards
