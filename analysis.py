"""체스 승률·다음 수 예측 계산부 (데스크톱 앱이 직접 불러 씀, HTTP 없음).

PGN 을 붙여 넣거나, 보드에서 직접 둔 수순(UCI 목록)을 받아 포지션마다 승률과 다음 수 분포를 계산합니다.
추론 규칙은 predict.py 와 같습니다 (chesswdl.inference.run_game, chesswdl.expected.move_weighted_wdl 재사용).
"""
import hashlib
import json
import os
import sys
import threading
import time
from collections import OrderedDict

import chess
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from chesswdl.checkpoint import load_checkpoint  # noqa: E402
from chesswdl.data import GameView, norm_name  # noqa: E402
from chesswdl.encoding import index_to_move  # noqa: E402
from chesswdl.expected import EngineValue, flip_note, move_weighted_wdl  # noqa: E402
from chesswdl.inference import game_batch, run_game  # noqa: E402
from chesswdl.pgn_extract import GameFilter, parse_pgn_string  # noqa: E402
from chesswdl.policy import DEFAULT_TOP_P  # noqa: E402

MAX_BODY = 2 * 1024 * 1024
ELO_MIN, ELO_MAX = 100, 3449          # Elo 임베딩은 50점 단위 70칸 (0 ~ 3450)
POLICY_TOP_K = 12                      # 정책 상위 몇 개까지 화면에 보낼지
MAX_PLIES = 600
CACHE_SIZE = 50000                     # 수별 가중 결과 캐시 (포지션 단위)
FILT = GameFilter(rated_only=False, min_est_sec=0, max_est_sec=1e12, require_elo=False,
                  min_plies=1, allow_unfinished=True)


class UserError(Exception):
    """사용자 입력 문제 → HTTP 400 으로 메시지 그대로 전달."""


def r4(x):
    return round(float(x), 4)


def white_view(wdl, white_to_move):
    """둘 차례 기준 [승, 무, 패] → 백 기준 {w: 백 승, d: 무, b: 흑 승}."""
    w, d, l = (float(v) for v in wdl)
    if white_to_move:
        return {"w": r4(w), "d": r4(d), "b": r4(l)}
    return {"w": r4(l), "d": r4(d), "b": r4(w)}


def white_view_range(rng, white_to_move):
    """(3, 2) [하한, 상한] 범위를 백 기준으로."""
    rows = [[r4(rng[i, 0]), r4(rng[i, 1])] for i in range(3)]
    if white_to_move:
        return {"w": rows[0], "d": rows[1], "b": rows[2]}
    return {"w": rows[2], "d": rows[1], "b": rows[0]}


def white_note(txt, white_to_move):
    if txt is None:
        return None
    return txt if white_to_move else flip_note(txt)


def diagnose(text):
    """parse_pgn_string 이 아무것도 못 돌려줬을 때 이유를 추정."""
    if not text.strip():
        return "PGN이 비어 있습니다. 기보를 붙여 넣으세요."
    if "[FEN " in text or '[SetUp "1"]' in text:
        return "시작 포지션이 표준이 아닌 대국(FEN 태그 포함)은 분석할 수 없습니다."
    if "[Variant " in text and '[Variant "Standard"]' not in text:
        return "표준 체스가 아닌 변형 대국은 분석할 수 없습니다."
    return "PGN을 읽지 못했습니다. 수 표기(SAN)에 잘못되었거나 둘 수 없는 수가 있는지 확인하세요."


def parse_elo(v, name):
    if v is None or v == "":
        return None
    try:
        e = int(v)
    except (TypeError, ValueError):
        raise UserError(f"{name} 레이팅은 숫자로 입력하세요.")
    if not ELO_MIN <= e <= ELO_MAX:
        raise UserError(f"{name} 레이팅은 {ELO_MIN}~{ELO_MAX} 사이로 입력하세요.")
    return e


def clean_tag(v, default="?"):
    v = str(v or "").strip().replace('"', "'").replace("\n", " ")[:80]
    return v or default


def pgn_from_moves(moves, white=None, black=None, time_control=None):
    """보드에서 직접 둔 수순(UCI 리스트) → PGN 문자열. 둘 수 없는 수가 있으면 UserError."""
    if not isinstance(moves, list):
        raise UserError("moves 는 UCI 수의 목록이어야 합니다. 예: [\"e2e4\", \"e7e5\"]")
    if len(moves) > MAX_PLIES:
        raise UserError(f"수순이 너무 깁니다 ({MAX_PLIES}수까지).")
    b, parts = chess.Board(), []
    for i, u in enumerate(moves):
        try:
            m = chess.Move.from_uci(str(u))
        except ValueError:
            m = None
        if m is None or m not in b.legal_moves:
            raise UserError(f"{i // 2 + 1}번째 수({u})는 이 포지션에서 둘 수 없습니다.")
        if b.is_game_over():
            raise UserError("이미 끝난 대국에 수를 더 둘 수 없습니다.")
        san = b.san(m)
        parts.append(f"{i // 2 + 1}. {san}" if i % 2 == 0 else san)
        b.push(m)
    head = f'[White "{clean_tag(white)}"]\n[Black "{clean_tag(black)}"]\n[Result "*"]\n'
    if time_control:
        head += f'[TimeControl "{clean_tag(time_control, "-")}"]\n'
    return head + "\n" + " ".join(parts + ["*"]) + "\n"


class Analyzer:
    def __init__(self, ckpt_path, device, engine_path=None, calib=None, depth=12, threads=1, hash_mb=64,
                 top_p=DEFAULT_TOP_P, latest=None):
        self.device = device
        self.model, self.vocab, self.ck = load_checkpoint(ckpt_path, device)
        self.model.eval()
        self.window = self.model.cfg["window"]
        self.ckpt_path = ckpt_path
        self.latest = latest or {}
        self.top_p = top_p
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.engine = None
        self.engine_value = None
        if engine_path:
            from chesswdl.engine import Engine
            self.engine = Engine(engine_path, depth=depth, threads=threads, hash_mb=hash_mb)
            if calib:
                self.engine_value = EngineValue(self.engine, calib)

    def info(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        return {
            "ckpt": os.path.relpath(self.ckpt_path, ROOT).replace("\\", "/"),
            "round": self.latest.get("round"),
            "months": self.latest.get("months", []),
            "device": str(self.device),
            "params_m": round(n_params / 1e6, 2),
            "window": self.window,
            "top_p": self.top_p,
            "engine": self.engine.name if self.engine else None,
            "weighted_modes": ["model"] + (["engine"] if self.engine_value else []),
            "players_in_vocab": len(self.vocab),
        }

    @torch.no_grad()
    def analyze(self, req):
        if req.get("moves") is not None:
            pgn = pgn_from_moves(req["moves"], req.get("white"), req.get("black"), req.get("time_control"))
        else:
            pgn = req.get("pgn") or ""
        if len(pgn) > MAX_BODY:
            raise UserError("PGN이 너무 깁니다. 대국 한 판만 붙여 넣으세요.")
        white_elo = parse_elo(req.get("white_elo"), "백")
        black_elo = parse_elo(req.get("black_elo"), "흑")
        weighted = req.get("weighted") or "model"
        if weighted not in ("off", "model", "engine"):
            raise UserError("weighted 는 off, model, engine 중 하나여야 합니다.")
        if weighted == "engine" and not self.engine_value:
            raise UserError("엔진 평가 방식은 서버를 --engine 과 --calib 로 실행했을 때만 쓸 수 있습니다.")
        top_p = float(req.get("top_p") or self.top_p)
        if not 0.0 < top_p <= 1.0:
            raise UserError("top_p 는 0보다 크고 1 이하여야 합니다.")
        use_players = req.get("use_players", True)

        games = parse_pgn_string(pgn, FILT, store_san=True, include_final=True)
        if not games:
            raise UserError(diagnose(pgn))
        g = games[0]
        h = g["headers"]
        elo_in_pgn = {"white": "WhiteElo" in h and str(h["WhiteElo"]).isdigit(),
                      "black": "BlackElo" in h and str(h["BlackElo"]).isdigit()}
        pgn_elo = {"white": int(g["white_elo"]), "black": int(g["black_elo"])}
        if white_elo is not None:
            g["white_elo"] = white_elo
        if black_elo is not None:
            g["black_elo"] = black_elo

        sans = list(g["sans"])
        n = len(g["moves"])                     # 모델이 예측하는 포지션 수 (진행 중이면 마지막 포지션 포함)
        boards, b = [], chess.Board()
        for t in range(len(sans) + 1):          # 화면에 그릴 모든 포지션
            boards.append(b.copy(stack=False))
            if t < len(sans):
                b.push_san(sans[t])

        t0 = time.time()
        with self.lock:
            evtxt = [None] * n
            if self.engine:
                for t in range(n):
                    ev, txt = self.engine.evaluate(boards[t])     # 둘 차례 기준
                    g["evals"][t], g["has_eval"][t], evtxt[t] = ev, 1, txt

            wid = self.vocab.get(norm_name(g["white"]), 0) if use_players else 0
            bid = self.vocab.get(norm_name(g["black"]), 0) if use_players else 0
            gv = GameView.from_record(g)
            batch, hh, direct, pol = run_game(self.model, game_batch(gv, self.window, wid, bid), self.device)

            # 수별 가중 결과 캐시 키: 대국 조건 + 그 포지션까지의 수순·시계·평가 (이력이 같으면 결과도 같음)
            hsh = hashlib.sha1(json.dumps([int(g["white_elo"]), int(g["black_elo"]), wid, bid, float(g["base"]),
                                           float(g["inc"]), top_p, weighted, self.window]).encode())
            cache_hits = 0
            positions = []
            for t in range(len(boards)):
                bd = boards[t]
                wtm = bd.turn == chess.WHITE
                if t > 0:
                    hsh.update(sans[t - 1].encode() + b"|")
                if t < n:
                    hsh.update(np.asarray(g["clocks"][t], np.float32).tobytes() + bytes([int(g["has_clock"][t])])
                               + np.float32(g["evals"][t]).tobytes() + bytes([int(g["has_eval"][t])]))
                pos = {
                    "ply": t,
                    "fen": bd.fen(),
                    "white_to_move": wtm,
                    "last_move": boards[t - 1].parse_san(sans[t - 1]).uci() if t > 0 else None,
                    "next_san": sans[t] if t < len(sans) else None,
                    "check": bd.is_check(),
                    "legal": [m.uci() for m in bd.legal_moves],
                    "terminal": None,
                    "model": None,
                }
                if bd.is_game_over():
                    oc = bd.outcome()
                    pos["terminal"] = oc.termination.name.lower() if oc else "game_over"
                if t >= n:
                    positions.append(pos)
                    continue

                pt = pol[t]
                n_legal = bd.legal_moves.count()
                k = min(POLICY_TOP_K, n_legal)
                vals, ids = pt.topk(k) if k > 0 else (torch.zeros(0), torch.zeros(0, dtype=torch.long))
                policy = []
                for v, ix in zip(vals.tolist(), ids.tolist()):
                    if v <= 0:
                        continue
                    m = index_to_move(bd, ix)
                    if m is not None:
                        policy.append({"san": bd.san(m), "uci": m.uci(), "p": r4(v)})

                played = None
                mi = int(g["moves"][t])
                if mi >= 0:
                    p_played = float(pt[mi])
                    played = {"san": sans[t], "p": r4(p_played),
                              "rank": int((pt > pt[mi]).sum()) + 1, "n_legal": n_legal}

                model_out = {
                    "direct": white_view(direct[t], wtm),
                    "policy": policy,
                    "played": played,
                    "weighted": None,
                    "eval": white_note(evtxt[t], wtm),
                }
                wkey = hsh.hexdigest()
                if weighted != "off" and not bd.is_game_over() and wkey in self.cache:
                    self.cache.move_to_end(wkey)
                    model_out["weighted"] = self.cache[wkey]
                    cache_hits += 1
                elif weighted != "off" and not bd.is_game_over():
                    raw = {"mover_elo": g["white_elo"] if wtm else g["black_elo"],
                           "opp_elo": g["black_elo"] if wtm else g["white_elo"],
                           "mover_clock": float(g["clocks"][t, 0]), "opp_clock": float(g["clocks"][t, 1]),
                           "has_clock": float(g["has_clock"][t])}
                    r = move_weighted_wdl(self.model, batch, hh, t, bd, pt, top_p, weighted,
                                          self.engine_value, raw, self.engine)
                    model_out["weighted"] = {
                        "conditional": white_view(r["conditional"], wtm),
                        "range": white_view_range(r["range"], wtm),
                        "selected_mass": r4(r["selected_mass"]),
                        "excluded_mass": r4(r["excluded_mass"]),
                        "n_legal": int(r["n_legal"]),
                        "candidates": [{"san": c["san"], "uci": c["uci"], "p": r4(c["p"]), "cum": r4(c["cum"]),
                                        "after": white_view(c["wdl"], wtm),
                                        "engine": white_note(c["engine"], wtm)} for c in r["candidates"]],
                    }
                    self.cache[wkey] = model_out["weighted"]
                    if len(self.cache) > CACHE_SIZE:
                        self.cache.popitem(last=False)
                pos["model"] = model_out
                positions.append(pos)

        tc = h.get("TimeControl")
        return {
            "meta": {
                "white": g["white"], "black": g["black"],
                "white_elo": int(g["white_elo"]), "black_elo": int(g["black_elo"]),
                "pgn_elo": pgn_elo, "elo_in_pgn": elo_in_pgn,
                "elo_overridden": {"white": white_elo is not None and white_elo != pgn_elo["white"],
                                   "black": black_elo is not None and black_elo != pgn_elo["black"]},
                "known_players": {"white": wid > 0, "black": bid > 0},
                "result": h.get("Result", "*"),
                "event": h.get("Event"), "date": h.get("UTCDate") or h.get("Date"),
                "time_control": tc if tc not in (None, "-", "?") else None,
                "has_clock": bool(g["has_clock"].any()) if len(g["has_clock"]) else False,
                "has_eval": bool(np.asarray(g["has_eval"]).any()),
                "games_in_pgn": len(games),
                "n_plies": len(sans),
                "weighted_mode": weighted,
                "top_p": top_p,
                "seconds": round(time.time() - t0, 2),
                "cached_positions": cache_hits,
            },
            "positions": positions,
        }


def resolve_ckpt(arg):
    """--ckpt 가 없으면 ckpt/LATEST.json 의 현재 모델."""
    latest = {}
    lp = os.path.join(ROOT, "ckpt", "LATEST.json")
    if os.path.exists(lp):
        with open(lp, encoding="utf-8") as f:
            latest = json.load(f)
    path = arg or latest.get("ckpt")
    if not path:
        raise SystemExit("체크포인트를 찾지 못했습니다. --ckpt 로 경로를 지정하세요 (ckpt/LATEST.json 없음).")
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    if not os.path.exists(path):
        raise SystemExit(f"체크포인트 파일이 없습니다: {path}\n"
                         "Drive의 chess/ckpt 에서 받아 같은 경로에 두거나 --ckpt 로 지정하세요.")
    if arg and latest.get("ckpt") and os.path.normpath(os.path.join(ROOT, latest["ckpt"])) != os.path.normpath(path):
        latest = {}   # LATEST 가 아닌 모델이면 라운드 정보는 표시하지 않음
    return path, latest
