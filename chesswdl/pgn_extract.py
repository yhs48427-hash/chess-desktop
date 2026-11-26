"""PGN → ply 단위 배열 추출. (Drive chess/chesswdl 의 추론용 사본: 학습용 코드는 뺌)"""
import calendar
import io
import re
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import chess
import chess.pgn
import numpy as np

from .encoding import (board_tokens, eval_to_unit, legal_move_indices, move_index, state_features,
                       DEFAULT_ELO)

CLK_RE = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")
EVAL_RE = re.compile(r"\[%eval\s+(#?)([+-]?\d+(?:\.\d+)?)")
SKIPPED = object()


@dataclass
class GameFilter:
    rated_only: bool = True
    min_est_sec: float = 180
    max_est_sec: float = 1499
    min_elo: int = 0
    max_elo: int = 4000
    require_elo: bool = True
    min_plies: int = 10
    allow_unfinished: bool = False
    bad_terminations: tuple = ("Abandoned", "Rules infraction")


def parse_time_control(tc: str):
    if not tc or tc in ("-", "?"):
        return None, None
    try:
        base, inc = tc.split("+")
        return float(base), float(inc)
    except ValueError:
        try:
            return float(tc), 0.0
        except ValueError:
            return None, None


def parse_timestamp(h: dict) -> int:
    date = h.get("UTCDate") or h.get("Date") or ""
    t = h.get("UTCTime") or h.get("StartTime") or "00:00:00"
    try:
        st = time.strptime(f"{date} {t}", "%Y.%m.%d %H:%M:%S")
        return calendar.timegm(st)
    except ValueError:
        return 0


def _int_or(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


class ExtractVisitor(chess.pgn.BaseVisitor):
    def __init__(self, filt: Optional[GameFilter] = None, store_san=False, include_final=False):
        self.filt = filt or GameFilter()
        self.store_san = store_san
        self.include_final = include_final

    def begin_game(self):
        self.h = {}
        self.skip = False
        self.boards, self.states, self.clocks, self.evals, self.has_eval = [], [], [], [], []
        self.moves, self.sans = [], []
        self.legal = []
        self.pending_eval = None
        self.saw_clock = False
        self.last_board = None

    def visit_header(self, tagname, tagvalue):
        self.h[tagname] = tagvalue

    def _accept(self) -> bool:
        f, h = self.filt, self.h
        if h.get("SetUp") == "1" or "FEN" in h:
            return False
        if h.get("Variant", "Standard") not in ("Standard", "Chess"):
            return False
        res = h.get("Result", "*")
        if res not in ("1-0", "0-1", "1/2-1/2") and not f.allow_unfinished:
            return False
        if h.get("Termination") in f.bad_terminations:
            return False
        if f.rated_only and "Rated" not in h.get("Event", ""):
            return False
        base, inc = parse_time_control(h.get("TimeControl", ""))
        if f.min_est_sec > 0 or f.max_est_sec < 1e9:
            if base is None:
                return False
            est = base + 40 * inc
            if not (f.min_est_sec <= est <= f.max_est_sec):
                return False
        we, be = _int_or(h.get("WhiteElo"), None), _int_or(h.get("BlackElo"), None)
        if f.require_elo and (we is None or be is None):
            return False
        for e in (we, be):
            if e is not None and not (f.min_elo <= e <= f.max_elo):
                return False
        return True

    def end_headers(self):
        if not self._accept():
            self.skip = True
            return chess.pgn.SKIP
        self.base, self.inc = parse_time_control(self.h.get("TimeControl", ""))
        b = self.base if self.base is not None else 0.0
        self.clock = [b, b]

    def begin_variation(self):
        return chess.pgn.SKIP

    def _record(self, board):
        ply = len(self.boards)
        mover_white = board.turn == chess.WHITE
        self.boards.append(board_tokens(board))
        self.states.append(state_features(board, ply))
        self.legal.append(legal_move_indices(board))
        me, opp = (0, 1) if mover_white else (1, 0)
        self.clocks.append((self.clock[me], self.clock[opp]))
        if self.pending_eval is not None:
            v = self.pending_eval if mover_white else -self.pending_eval
            self.evals.append(v)
            self.has_eval.append(1)
        else:
            self.evals.append(0.0)
            self.has_eval.append(0)
        self.pending_eval = None

    def visit_move(self, board, move):
        if self.skip:
            return
        self._record(board)
        self.moves.append(move_index(move, board.turn))
        if self.store_san:
            self.sans.append(board.san(move))

    def visit_board(self, board):
        self.last_board = board

    def visit_comment(self, comment):
        if self.skip or not self.moves:
            return
        just_moved_white = (len(self.moves) - 1) % 2 == 0
        m = CLK_RE.search(comment)
        if m:
            sec = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            self.clock[0 if just_moved_white else 1] = sec
            self.saw_clock = True
        m = EVAL_RE.search(comment)
        if m:
            self.pending_eval = eval_to_unit(float(m.group(2)), m.group(1) == "#")

    def handle_error(self, error):
        self.skip = True

    def result(self):
        if self.skip:
            return SKIPPED
        if self.include_final and self.last_board is not None and not self.last_board.is_game_over():
            self._record(self.last_board)
            self.moves.append(-1)
        n = len(self.boards)
        if n < self.filt.min_plies:
            return SKIPPED
        res = self.h.get("Result", "*")
        white_score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(res)
        ply = np.arange(n)
        mover_white = ply % 2 == 0
        if white_score is None:
            wdl = np.full(n, 255, dtype=np.uint8)
        elif white_score == 0.5:
            wdl = np.ones(n, dtype=np.uint8)
        else:
            mover_won = mover_white == (white_score == 1.0)
            wdl = np.where(mover_won, 0, 2).astype(np.uint8)
        clocks = np.asarray(self.clocks, dtype=np.float32)
        has_clock = np.full(n, 1 if self.saw_clock else 0, dtype=np.uint8)
        if not self.saw_clock:
            clocks[:] = 0.0
        return {
            "boards": np.stack(self.boards),
            "state": np.stack(self.states),
            "clocks": clocks,
            "has_clock": has_clock,
            "evals": np.asarray(self.evals, dtype=np.float32),
            "has_eval": np.asarray(self.has_eval, dtype=np.uint8),
            "moves": np.asarray(self.moves, dtype=np.int16),
            "wdl": wdl,
            "legal": self.legal,
            "legal_len": np.asarray([len(x) for x in self.legal], dtype=np.uint8),
            "white": self.h.get("White", "?"),
            "black": self.h.get("Black", "?"),
            "white_elo": _int_or(self.h.get("WhiteElo"), DEFAULT_ELO),
            "black_elo": _int_or(self.h.get("BlackElo"), DEFAULT_ELO),
            "base": float(self.base or 0.0),
            "inc": float(self.inc or 0.0),
            "timestamp": parse_timestamp(self.h),
            "game_id": self.h.get("Site", self.h.get("Link", "")),
            "sans": self.sans,
            "headers": self.h,
        }


def iter_games(handle, filt: GameFilter, **kw) -> Iterator[dict]:
    while True:
        g = chess.pgn.read_game(handle, Visitor=lambda: ExtractVisitor(filt, **kw))
        if g is None:
            return
        if g is SKIPPED:
            continue
        yield g


def parse_pgn_string(text: str, filt: GameFilter, **kw):
    return list(iter_games(io.StringIO(text), filt, **kw))


PLY_KEYS = ["boards", "state", "clocks", "has_clock", "evals", "has_eval", "moves", "wdl", "legal_len"]
