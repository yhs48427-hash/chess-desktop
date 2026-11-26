"""Stockfish 연동. (Drive chess/chesswdl 의 추론용 사본)"""
import chess
import chess.engine

from .encoding import eval_to_unit


class Engine:
    def __init__(self, path, depth=12, movetime=None, threads=1, hash_mb=64):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Threads": threads, "Hash": hash_mb})
        self.limit = chess.engine.Limit(time=movetime) if movetime else chess.engine.Limit(depth=depth)
        self.name = self.engine.id.get("name", "engine")
        self.cache = {}

    def evaluate(self, board: chess.Board):
        key = board.fen()
        if key in self.cache:
            return self.cache[key]
        if board.is_checkmate():
            res = (-1.0, "#-0")
        elif board.is_stalemate() or board.is_insufficient_material():
            res = (0.0, "0.00")
        else:
            info = self.engine.analyse(board, self.limit)
            sc = info["score"].pov(board.turn)
            if sc.is_mate():
                m = sc.mate()
                res = (eval_to_unit(1.0 if m > 0 else -1.0, True), f"#{m}")
            else:
                cp = sc.score()
                res = (eval_to_unit(cp / 100.0, False), f"{cp / 100.0:+.2f}")
        self.cache[key] = res
        return res

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
