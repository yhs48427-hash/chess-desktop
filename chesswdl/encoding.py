"""보드/수/메타 정보를 정수·실수 텐서로 바꾸는 규칙. (Drive chess/chesswdl 의 추론용 사본)"""
import math

import chess
import numpy as np

NUM_PIECE_TOKENS = 13
NUM_CASTLE = 16
NUM_EP = 9
HALFMOVE_MAX = 100
PLY_BUCKETS = 64
ELO_BUCKETS = 70
POLICY_SIZE = 4096 + 72
N_CONT = 7

MAX_CLOCK_SEC = 10800.0
DEFAULT_ELO = 1500


def orient(square: int, turn: bool) -> int:
    return square if turn == chess.WHITE else chess.square_mirror(square)


def board_tokens(board: chess.Board) -> np.ndarray:
    us = board.turn
    out = np.zeros(64, dtype=np.uint8)
    for sq, pc in board.piece_map().items():
        out[orient(sq, us)] = pc.piece_type + (0 if pc.color == us else 6)
    return out


def state_features(board: chess.Board, ply: int) -> np.ndarray:
    us, them = board.turn, not board.turn
    castle = (int(board.has_kingside_castling_rights(us))
              | int(board.has_queenside_castling_rights(us)) << 1
              | int(board.has_kingside_castling_rights(them)) << 2
              | int(board.has_queenside_castling_rights(them)) << 3)
    ep = 0
    if board.ep_square is not None and board.has_legal_en_passant():
        ep = chess.square_file(board.ep_square) + 1
    hm = min(board.halfmove_clock, HALFMOVE_MAX)
    plyb = min(ply // 4, PLY_BUCKETS - 1)
    return np.array([castle, ep, hm, plyb], dtype=np.int16)


_UNDER = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}


def move_index(move: chess.Move, turn: bool) -> int:
    f = orient(move.from_square, turn)
    t = orient(move.to_square, turn)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        d = chess.square_file(t) - chess.square_file(f) + 1
        return 4096 + (chess.square_file(f) * 3 + d) * 3 + _UNDER[move.promotion]
    return f * 64 + t


def legal_move_indices(board: chess.Board) -> np.ndarray:
    return np.fromiter((move_index(m, board.turn) for m in board.legal_moves), dtype=np.int16)


def index_to_move(board: chess.Board, idx: int):
    for m in board.legal_moves:
        if move_index(m, board.turn) == idx:
            return m
    return None


def elo_bucket(elo) -> np.ndarray:
    return np.clip(np.asarray(elo) // 50, 0, ELO_BUCKETS - 1).astype(np.int64)


def log_clock(sec) -> np.ndarray:
    sec = np.maximum(np.asarray(sec, dtype=np.float32), 0.0)
    return np.log1p(sec) / math.log1p(MAX_CLOCK_SEC)


def eval_to_unit(pawns_white: float, is_mate: bool) -> float:
    if is_mate:
        return 1.0 if pawns_white > 0 else -1.0
    return math.tanh(pawns_white / 4.0)


MATERIAL = np.array([0, 1, 3, 3, 5, 9, 0, -1, -3, -3, -5, -9, 0], dtype=np.float32)
