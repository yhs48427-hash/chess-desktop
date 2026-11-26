"""체스 승률 예측 데스크톱 앱 (PySide6). 서버나 브라우저 없이 모델을 같은 프로세스에서 직접 부릅니다.

    run.bat                      # 보통은 이걸로 실행 (start.py 가 환경과 모델 파일을 준비한 뒤 이 파일을 실행)
    python app.py [--ckpt ckpt/round03_2025-03.pt] [--device cpu] [--engine stockfish.exe --calib results/calib.json]

기능: PGN 붙여넣기 분석, 보드에서 끌어서/눌러서 두기, 승격, 기보에서 갈라지기와 되돌아가기,
     기준 레이팅과 비교 레이팅, 후보 수 표(눌러서 두기), 대국 전체 승률 그래프, 기보 목록.
계산은 analysis.Analyzer 가 하고(웹 버전 server.py 와 같은 코드), 화면이 멈추지 않도록 별도 스레드에서 돌립니다.
"""
import argparse
import html
import math
import os
import sys
import threading
import traceback

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetricsF, QIntValidator, QKeySequence, QPainter,
                           QPainterPath, QPen, QPolygonF, QShortcut)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMainWindow, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy,
                               QVBoxLayout, QWidget)

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ------------------------------------------------------------------ 색과 글꼴
PAPER, SURFACE, INK, INK2, LINE = "#E9ECE8", "#F4F5F2", "#1C2326", "#4A5559", "#C3CAC6"
SQ_L, SQ_D = "#D5DCE3", "#8094A8"
W_WIN, DRAW, B_WIN = "#FAF8F2", "#A9B1B5", "#242C30"
MARK, COBALT, ERROR = "#F0A202", "#2F4BC0", "#A8261C"
SANS, SERIF = "Gothic A1", "Hahmlet"
GLYPH_FONTS = ["Segoe UI Symbol", "Apple Symbols", "DejaVu Sans", "Noto Sans Symbols 2"]
GLYPH = {"k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}
PROMO_NAME = {"q": "퀸", "r": "룩", "b": "비숍", "n": "나이트"}
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
TERM = {"stalemate": "스테일메이트로 무승부입니다.", "insufficient_material": "기물이 부족해 무승부입니다.",
        "seventyfive_moves": "75수 규칙으로 무승부입니다.", "fivefold_repetition": "같은 포지션이 다섯 번 나와 무승부입니다."}
SAMPLE = """[Event "Rated Blitz game"]
[Site "https://lichess.org/example"]
[Date "2025.04.12"]
[White "Seoul_Knight"]
[Black "Hangang_Rook"]
[Result "1-0"]
[WhiteElo "1712"]
[BlackElo "1748"]
[TimeControl "180+2"]

1. e4 { [%clk 0:03:00] } 1... e5 { [%clk 0:03:00] } 2. Nf3 { [%clk 0:02:59] } 2... Nc6 { [%clk 0:02:58] } 3. Bc4 { [%clk 0:02:57] } 3... Bc5 { [%clk 0:02:55] } 4. c3 { [%clk 0:02:55] } 4... Nf6 { [%clk 0:02:51] } 5. d4 { [%clk 0:02:50] } 5... exd4 { [%clk 0:02:47] } 6. cxd4 { [%clk 0:02:48] } 6... Bb4+ { [%clk 0:02:44] } 7. Bd2 { [%clk 0:02:45] } 7... Bxd2+ { [%clk 0:02:40] } 8. Nbxd2 { [%clk 0:02:43] } 8... d5 { [%clk 0:02:31] } 9. exd5 { [%clk 0:02:40] } 9... Nxd5 { [%clk 0:02:29] } 10. Qb3 { [%clk 0:02:33] } 10... Nce7 { [%clk 0:02:18] } 11. O-O { [%clk 0:02:30] } 11... O-O { [%clk 0:02:15] } 12. Rfe1 { [%clk 0:02:24] } 12... c6 { [%clk 0:02:06] } 13. Ne4 { [%clk 0:02:15] } 13... Nf5 { [%clk 0:01:58] } 14. Neg5 { [%clk 0:02:05] } 14... h6 { [%clk 0:01:50] } 15. Nxf7 { [%clk 0:01:58] } 15... Rxf7 { [%clk 0:01:41] } 16. Bxd5 { [%clk 0:01:55] } 16... cxd5 { [%clk 0:01:37] } 17. Qxd5 { [%clk 0:01:52] } 17... Qf6 { [%clk 0:01:25] } 18. Re8+ { [%clk 0:01:47] } 18... Kh7 { [%clk 0:01:22] } 19. Qe4 { [%clk 0:01:38] } 19... Rf8 { [%clk 0:01:10] } 20. Rxf8 { [%clk 0:01:33] } 1-0"""


def qc(hex_or_rgba, alpha=None):
    c = QColor(hex_or_rgba)
    if alpha is not None:
        c.setAlphaF(alpha)
    return c


def font(family, px, weight=400, italic=False):
    f = QFont(family)
    f.setPixelSize(int(px))
    f.setWeight(QFont.Weight(weight))
    f.setItalic(italic)
    return f


def load_fonts():
    for name in ("GothicA1-Regular.ttf", "GothicA1-Medium.ttf", "GothicA1-Bold.ttf", "Hahmlet-Variable.ttf"):
        p = os.path.join(ROOT, "fonts", name)
        if os.path.exists(p):
            QFontDatabase.addApplicationFont(p)


def glyph_family():
    fams = set(QFontDatabase.families())
    for f in GLYPH_FONTS:
        if f in fams:
            return f
    return QFont().defaultFamily()


# ------------------------------------------------------------------ 계산 도우미
def pct(v):
    x = v * 100
    if 0 < x < 1:
        return "<1"
    if 99 < x < 100:
        return ">99"
    return str(round(x))


def pct1(v):
    x = v * 100
    if 0 < x < 0.1:
        return "<0.1"
    return f"{x:.1f}" if x < 10 else str(round(x))


def move_label(t, san):
    n = t // 2 + 1
    return (f"{n}. " if t % 2 == 0 else f"{n}... ") + san


def side(white):
    return "백" if white else "흑"


def eng_txt(t):
    if not t:
        return ""
    return "0.00" if t in ("+0.00", "-0.00") else t


def sq_index(name):
    return (ord(name[1]) - 49) * 8 + (ord(name[0]) - 97)


def parse_fen(fen):
    out = [None] * 64
    for i, row in enumerate(fen.split(" ")[0].split("/")):
        f, r = 0, 7 - i
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                out[r * 8 + f] = ("w" if ch.isupper() else "b", ch.lower())
                f += 1
    return out


def terminal_wdl(pos):
    if pos["terminal"] == "checkmate":
        return {"w": 0, "d": 0, "b": 1} if pos["white_to_move"] else {"w": 1, "d": 0, "b": 0}
    return {"w": 0, "d": 1, "b": 0}


def wdl_of(pos, mode="weighted"):
    if not pos:
        return None
    m = pos.get("model")
    if m:
        if mode == "weighted" and m.get("weighted"):
            return m["weighted"]["conditional"]
        return m["direct"]
    return terminal_wdl(pos) if pos.get("terminal") else None


def legal_map(pos):
    out = {}
    for u in (pos or {}).get("legal") or []:
        out.setdefault(sq_index(u[:2]), []).append((sq_index(u[2:4]), u))
    return out


class State:
    def __init__(self):
        self.info = None
        self.line = None      # {"kind": "pgn", "pgn": ...} 또는 {"kind": "moves", "moves": [...], "white", "black", "tc"}
        self.eloA = None      # {"w", "b"}  기준 레이팅 (None 이면 PGN 값)
        self.eloB = None      # 비교 레이팅 또는 None
        self.orig = None      # 붙여 넣은 기보에서 갈라졌을 때 돌아갈 곳 {"line", "ply", "moves"}
        self.A = self.B = None
        self.ply = 0
        self.flip = False
        self.curve = "weighted"
        self.arrows = "B"
        self.show_played = True
        self.busy = False
        self.show_all = False
        self.sel = None
        self.drag = None      # {"from", "pos", "moved", "toggle"}


# ------------------------------------------------------------------ 보드
def arrow_poly(a, b, w):
    dx, dy = b.x() - a.x(), b.y() - a.y()
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    head, hw = max(30, w * 1.9), max(17, w * 1.2)
    sx, sy = a.x() + ux * 20, a.y() + uy * 20
    ex, ey = b.x() - ux * 8, b.y() - uy * 8
    bx, by = ex - ux * head, ey - uy * head
    pts = [(sx + px * w / 2, sy + py * w / 2), (bx + px * w / 2, by + py * w / 2), (bx + px * hw, by + py * hw), (ex, ey),
           (bx - px * hw, by - py * hw), (bx - px * w / 2, by - py * w / 2), (sx - px * w / 2, sy - py * w / 2)]
    return QPolygonF([QPointF(x, y) for x, y in pts])


class BoardWidget(QWidget):
    """보드 + 왼쪽 세로 승률 막대. 내부 좌표는 보드 한 변 = 800 단위 (웹 버전과 같은 기하)."""
    BAR = 34   # 세로 막대 영역 폭 (px)

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)    # 보드를 누르면 글 입력칸에서 포커스가 빠져 ← → 가 동작
        self.setMinimumSize(380, 340)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._glyphs = {}
        self.busy_text = False

    # 좌표 -------------------------------------------------------
    def geom(self):
        s = max(100.0, min(self.width() - self.BAR, self.height()))
        ox = self.BAR + (self.width() - self.BAR - s) / 2
        oy = (self.height() - s) / 2
        return ox, oy, s

    def center(self, idx):
        S = self.win.S
        f, r = idx % 8, idx // 8
        fx = 7 - f if S.flip else f
        fy = r if S.flip else 7 - r
        return QPointF(fx * 100 + 50, fy * 100 + 50)

    def to_board(self, pos):
        ox, oy, s = self.geom()
        return QPointF((pos.x() - ox) / s * 800, (pos.y() - oy) / s * 800)

    def square_at(self, pos):
        p = self.to_board(pos)
        if not (0 <= p.x() < 800 and 0 <= p.y() < 800):
            return -1
        fx, fy = int(p.x() // 100), int(p.y() // 100)
        S = self.win.S
        return (fy if S.flip else 7 - fy) * 8 + (7 - fx if S.flip else fx)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        # 보드 열 폭을 '높이 + 막대 폭'으로 제한 → 이름표와 버튼 줄이 보드 폭에 맞음 (폭에서 다시 계산하지 않아 되먹임 없음)
        self.win.board_col.setMaximumWidth(self.height() + self.BAR)

    def glyph_path(self, t):
        if t not in self._glyphs:
            path = QPainterPath()
            path.addText(0, 0, font(glyph_family(), 78), GLYPH[t] + "︎")
            br = path.boundingRect()
            path.translate(-br.center().x(), -br.center().y())
            self._glyphs[t] = path
        return self._glyphs[t]

    # 그리기 -----------------------------------------------------
    def paintEvent(self, _):
        win, S = self.win, self.win.S
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        ox, oy, s = self.geom()
        # 세로 승률 막대 (A: 굵게, B: 파랑 테두리)
        self.draw_vbar(p, QRectF(ox - self.BAR, oy, 14, s), wdl_of(win.cur_pos(S.A)), INK)
        if S.B:
            self.draw_vbar(p, QRectF(ox - self.BAR + 19, oy, 8, s), wdl_of(win.cur_pos(S.B)), COBALT)

        p.save()
        p.translate(ox, oy)
        k = s / 800
        p.scale(k, k)
        pos = win.cur_pos(S.A)
        sq = parse_fen(pos["fen"] if pos else START_FEN)
        coord_font = font(SANS, 15, 700)
        for idx in range(64):
            f, r = idx % 8, idx // 8
            c = self.center(idx)
            light = (f + r) % 2 == 1
            p.fillRect(QRectF(c.x() - 50, c.y() - 50, 100, 100), qc(SQ_L if light else SQ_D))
            col = qc(SQ_D if light else SQ_L)
            p.setPen(col)
            p.setFont(coord_font)
            if (7 - f if S.flip else f) == 0:
                p.drawText(QPointF(c.x() - 45, c.y() - 30), str(r + 1))
            if (r if S.flip else 7 - r) == 7:
                p.drawText(QPointF(c.x() + 36, c.y() + 46), "abcdefgh"[f])

        def tint(idx, color):
            c = self.center(idx)
            p.fillRect(QRectF(c.x() - 50, c.y() - 50, 100, 100), color)

        if pos and pos.get("last_move"):
            tint(sq_index(pos["last_move"][:2]), qc("#FFFFFF", .38))
            tint(sq_index(pos["last_move"][2:4]), qc("#FFFFFF", .38))
        if pos and pos.get("check"):
            me = "w" if pos["white_to_move"] else "b"
            for i, pc in enumerate(sq):
                if pc and pc[1] == "k" and pc[0] == me:
                    tint(i, qc(ERROR, .45))
        if S.sel is not None:
            c = self.center(S.sel)
            p.setPen(QPen(qc(INK), 6))
            p.setBrush(qc("#FFFFFF", .45))
            p.drawRect(QRectF(c.x() - 47, c.y() - 47, 94, 94))

        dragging = S.drag and S.drag["moved"]
        for i, pc in enumerate(sq):
            if pc and not (dragging and i == S.drag["from"]):
                self.draw_piece(p, pc, self.center(i))

        # 후보 수 화살표
        run = S.B if (S.B and S.arrows == "B") else S.A
        rp = win.cur_pos(run)
        if rp and rp.get("model") and S.sel is None:
            color = COBALT if run is S.B else MARK
            m = rp["model"]
            lst = [c for c in (m["weighted"]["candidates"] if m.get("weighted") else m["policy"]) if c["p"] >= 0.02][:8]
            maxp = max([c["p"] for c in lst] + [0.01])
            for c in reversed(lst):
                a, b = self.center(sq_index(c["uci"][:2])), self.center(sq_index(c["uci"][2:4]))
                w = min(34, 6 + 30 * math.sqrt(c["p"]))
                p.setPen(QPen(qc(W_WIN, .85), 2))
                p.setBrush(qc(color, 0.42 + 0.5 * c["p"] / maxp))
                p.drawPolygon(arrow_poly(a, b, w))
            p.setFont(font(SANS, 19, 700))
            for c in lst:
                if c["p"] < 0.08:
                    continue
                a, b = self.center(sq_index(c["uci"][:2])), self.center(sq_index(c["uci"][2:4]))
                mid = QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
                p.setPen(QPen(qc(color), 3))
                p.setBrush(qc(W_WIN))
                p.drawEllipse(mid, 21, 21)
                p.setPen(qc(INK))
                p.drawText(QRectF(mid.x() - 21, mid.y() - 21, 42, 42), Qt.AlignmentFlag.AlignCenter, pct(c["p"]))
        # 실제로 둔 다음 수 (점선)
        if S.show_played and S.sel is None and S.A and S.ply + 1 < len(S.A["positions"]):
            mv = S.A["positions"][S.ply + 1]["last_move"]
            pen = QPen(qc(INK), 3)
            pen.setDashPattern([3, 2.3])
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolygon(arrow_poly(self.center(sq_index(mv[:2])), self.center(sq_index(mv[2:4])), 12))
        # 고른 말이 갈 수 있는 칸
        if S.sel is not None:
            seen = set()
            mover = sq[S.sel]
            for to, _u in legal_map(pos).get(S.sel, []):
                if to in seen:
                    continue
                seen.add(to)
                c = self.center(to)
                capture = sq[to] is not None or (mover and mover[1] == "p" and (to - S.sel) % 8 != 0)
                if capture:
                    p.setPen(QPen(qc(INK, .45), 7))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawEllipse(c, 44, 44)
                else:
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(qc(INK, .42))
                    p.drawEllipse(c, 14, 14)
        # 끌고 있는 말
        if dragging:
            self.draw_piece(p, sq[S.drag["from"]], S.drag["pos"], lift=True)
        p.setPen(QPen(qc(INK, .35), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(0, 0, 800, 800))
        if self.busy_text:
            p.fillRect(QRectF(0, 0, 800, 800), qc(PAPER, .55))
            p.setPen(qc(INK))
            p.setFont(font(SERIF, 30, 600))
            p.drawText(QRectF(0, 0, 800, 800), Qt.AlignmentFlag.AlignCenter, "계산 중")
        p.restore()

    def draw_piece(self, p, pc, at, lift=False):
        color, t = pc
        path = self.glyph_path(t)
        p.save()
        p.translate(at)
        if lift:
            p.scale(1.08, 1.08)
        if color == "w":
            p.strokePath(path, QPen(qc(INK), 6.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.fillPath(path, qc(W_WIN))
        else:
            p.strokePath(path, QPen(qc(W_WIN), 2.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.fillPath(path, qc(INK))
        p.restore()

    def draw_vbar(self, p, r, wdl, border):
        W = wdl or {"w": 0, "d": 1, "b": 0}
        flip = self.win.S.flip
        top, bot = (W["w"], W["b"]) if flip else (W["b"], W["w"])
        tc, bc = (W_WIN, B_WIN) if flip else (B_WIN, W_WIN)
        h1, h2 = r.height() * top, r.height() * W["d"]
        p.fillRect(QRectF(r.x(), r.y(), r.width(), h1), qc(tc))
        p.fillRect(QRectF(r.x(), r.y() + h1, r.width(), h2), qc(DRAW))
        p.fillRect(QRectF(r.x(), r.y() + h1 + h2, r.width(), r.height() - h1 - h2), qc(bc))
        p.setPen(QPen(qc(border), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)

    # 마우스 ----------------------------------------------------
    def mousePressEvent(self, e):
        win, S = self.win, self.win.S
        if e.button() != Qt.MouseButton.LeftButton or not win.can_play():
            return
        sq = self.square_at(e.position())
        if sq < 0:
            return
        lm = legal_map(win.cur_pos(S.A))
        if S.sel is not None and any(to == sq for to, _ in lm.get(S.sel, [])):
            win.try_move(S.sel, sq)
            return
        if sq in lm:
            S.drag = {"from": sq, "pos": self.to_board(e.position()), "moved": False, "toggle": S.sel == sq}
            S.sel = sq
        else:
            S.sel = None
        self.update()

    def mouseMoveEvent(self, e):
        S = self.win.S
        if S.drag:
            p = self.to_board(e.position())
            c = self.center(S.drag["from"])
            if S.drag["moved"] or math.hypot(p.x() - c.x(), p.y() - c.y()) > 12:
                S.drag["moved"] = True
                S.drag["pos"] = p
                self.update()
            return
        sq = self.square_at(e.position())
        lm = legal_map(self.win.cur_pos(S.A)) if self.win.can_play() else {}
        targets = {to for to, _ in lm.get(S.sel, [])} if S.sel is not None else set()
        hand = sq in lm or sq in targets
        self.setCursor(Qt.CursorShape.PointingHandCursor if hand else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, e):
        win, S = self.win, self.win.S
        d = S.drag
        if not d:
            return
        S.drag = None
        sq = self.square_at(e.position())
        if d["moved"]:
            if sq >= 0 and sq != d["from"] and any(to == sq for to, _ in legal_map(win.cur_pos(S.A)).get(d["from"], [])):
                win.try_move(d["from"], sq)
                return
        elif d["toggle"]:
            S.sel = None
        self.update()


class PromoDialog(QDialog):
    def __init__(self, parent, white, ucis):
        super().__init__(parent)
        self.setWindowTitle("승격")
        self.choice = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 14)
        lay.setSpacing(12)
        t = QLabel("어떤 기물로 승격할까요?")
        t.setProperty("role", "h2")
        lay.addWidget(t, alignment=Qt.AlignmentFlag.AlignCenter)
        row = QHBoxLayout()
        for k in "qrbn":
            u = next((x for x in ucis if x.endswith(k)), None)
            if not u:
                continue
            # 백은 속이 빈 기호(♕), 흑은 채워진 기호(♛)
            b = QPushButton(chr(ord(GLYPH[k]) - 6) if white else GLYPH[k])
            b.setToolTip(PROMO_NAME[k])
            b.setAccessibleName(PROMO_NAME[k])
            b.setFixedSize(64, 64)
            b.setStyleSheet(f"QPushButton{{background:{SQ_L};border:1px solid {LINE};color:{INK};padding:0;"
                            f"font-family:'{glyph_family()}';font-size:44px;font-weight:400}}"
                            f"QPushButton:hover{{background:{SQ_D}}}")
            b.clicked.connect(lambda _=False, u=u: self._pick(u))
            row.addWidget(b)
        lay.addLayout(row)
        c = QPushButton("취소")
        c.setProperty("kind", "quiet")
        c.clicked.connect(self.reject)
        lay.addWidget(c, alignment=Qt.AlignmentFlag.AlignCenter)

    def _pick(self, u):
        self.choice = u
        self.accept()


# ------------------------------------------------------------------ 오른쪽 패널 부품
class HBar(QWidget):
    def __init__(self, wdl, slim=False, cobalt=False):
        super().__init__()
        self.w, self.cobalt = wdl, cobalt
        self.setFixedHeight(34 if slim else 46)
        self.slim = slim
        self.setAccessibleName(f"백 승 {pct(wdl['w'])}%, 무승부 {pct(wdl['d'])}%, 흑 승 {pct(wdl['b'])}%")

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        x = r.x()
        big, small = font(SERIF, 17 if self.slim else 20, 600), font(SANS, 11, 500)
        for key, name, bg, fg in (("w", "백", W_WIN, INK), ("d", "무", DRAW, INK), ("b", "흑", B_WIN, W_WIN)):
            v = self.w[key]
            seg = QRectF(x, r.y(), r.width() * v, r.height())
            p.fillRect(seg, qc(bg))
            if v >= 0.07:
                p.setPen(qc(fg))
                num = pct(v)
                if v >= 0.15:
                    fm1, fm2 = QFontMetricsF(small), QFontMetricsF(big)
                    tw = fm1.horizontalAdvance(name) + 5 + fm2.horizontalAdvance(num)
                    sx = seg.center().x() - tw / 2
                    p.setFont(small)
                    p.drawText(QRectF(sx, seg.y(), fm1.horizontalAdvance(name) + 2, seg.height()),
                               Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, name)
                    p.setFont(big)
                    p.drawText(QRectF(sx + fm1.horizontalAdvance(name) + 5, seg.y(), fm2.horizontalAdvance(num) + 4, seg.height()),
                               Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, num)
                else:
                    p.setFont(big)
                    p.drawText(seg, Qt.AlignmentFlag.AlignCenter, num)
            x += seg.width()
        p.setPen(QPen(qc(COBALT if self.cobalt else INK), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
        if self.cobalt:
            p.fillRect(QRectF(r.x(), r.y(), 4, r.height()), qc(COBALT))


class CandList(QWidget):
    """후보 수 표. 행을 누르면 그 수를 둠."""

    def __init__(self, rows, mode, on_play, mover="백", has_after=True, has_eng=False, played=None):
        super().__init__()
        self.rows, self.mode, self.on_play = rows, mode, on_play
        self.mover, self.has_after, self.has_eng, self.played = mover, has_after, has_eng, played
        self.row_h = 42 if mode == "compare" else 30
        self.head_h = 24
        self.hover = -1
        self.setMouseTracking(True)
        self.setFixedHeight(self.head_h + self.row_h * len(rows) + 2)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("수를 누르면 보드에 둡니다")

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W = self.width()
        p.setFont(font(SANS, 11, 500))
        p.setPen(qc(INK2))
        right = W - 4
        if self.mode == "compare":
            p.drawText(QRectF(4, 0, 80, self.head_h), Qt.AlignmentFlag.AlignVCenter, "수")
            p.drawText(QRectF(84, 0, 60, self.head_h), Qt.AlignmentFlag.AlignVCenter, "기준")
            p.setPen(qc(COBALT))
            p.drawText(QRectF(116, 0, 120, self.head_h), Qt.AlignmentFlag.AlignVCenter, "/ 비교 레이팅")
            p.setPen(qc(INK2))
            p.drawText(QRectF(right - 70, 0, 70, self.head_h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, "차이")
        else:
            p.drawText(QRectF(4, 0, 80, self.head_h), Qt.AlignmentFlag.AlignVCenter, "수")
            p.drawText(QRectF(84, 0, 100, self.head_h), Qt.AlignmentFlag.AlignVCenter, "고를 확률")
            if self.has_eng:
                p.drawText(QRectF(right - 56, 0, 56, self.head_h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, "엔진")
            if self.has_after:
                rr = right - (64 if self.has_eng else 0)
                p.drawText(QRectF(rr - 110, 0, 110, self.head_h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                           f"둔 뒤 {self.mover} 승률")
        p.setPen(QPen(qc(LINE), 1))
        p.drawLine(QPointF(0, self.head_h - .5), QPointF(W, self.head_h - .5))

        vals = [max(r.get("a") or 0, r.get("b") or 0) if self.mode == "compare" else r["p"] for r in self.rows]
        maxp = max(vals + [0.01])
        bar_max = (90 if self.mode == "compare" else (64 if self.has_eng else 110))
        for i, r in enumerate(self.rows):
            y = self.head_h + i * self.row_h
            row = QRectF(0, y, W, self.row_h)
            if i == self.hover:
                p.fillRect(row, qc(INK, .06))
            if r["san"] == self.played:
                p.fillRect(QRectF(0, y + 3, 3, self.row_h - 6), qc(INK))
            p.setPen(qc(INK))
            p.setFont(font(SERIF, 15, 600))
            p.drawText(QRectF(10, y, 74, self.row_h), Qt.AlignmentFlag.AlignVCenter, r["san"])
            p.setFont(font(SANS, 13))
            if self.mode == "compare":
                for j, (key, col) in enumerate((("a", MARK), ("b", COBALT))):
                    v = r.get(key)
                    yy = y + 6 + j * 16
                    if v is None:
                        p.setPen(qc(INK2))
                        p.drawText(QRectF(84, yy - 2, 120, 16), Qt.AlignmentFlag.AlignVCenter, "12위 밖")
                        continue
                    bw = max(2, bar_max * v / maxp)
                    p.fillRect(QRectF(84, yy + 3, bw, 8), qc(col))
                    p.setPen(qc(INK))
                    p.drawText(QRectF(84 + bw + 6, yy - 2, 60, 16), Qt.AlignmentFlag.AlignVCenter, pct1(v) + "%")
                if r.get("a") is not None and r.get("b") is not None:
                    d = (r["b"] - r["a"]) * 100
                    up = d >= 0.05
                    p.setPen(qc(COBALT if up else INK2))
                    p.setFont(font(SANS, 13, 700 if up else 400))
                    p.drawText(QRectF(right - 80, y, 80, self.row_h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                               f"{'+' if d >= 0 else ''}{d:.1f}%p")
            else:
                bw = max(2, bar_max * r["p"] / maxp)
                p.fillRect(QRectF(84, y + self.row_h / 2 - 4, bw, 8), qc(MARK))
                p.setPen(qc(INK))
                p.drawText(QRectF(84 + bw + 8, y, 60, self.row_h), Qt.AlignmentFlag.AlignVCenter, pct1(r["p"]) + "%")
                if self.has_eng:
                    p.drawText(QRectF(right - 56, y, 56, self.row_h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                               eng_txt(r.get("engine")))
                if self.has_after and r.get("after_mover") is not None:
                    rr = right - (64 if self.has_eng else 0)
                    p.drawText(QRectF(rr - 60, y, 60, self.row_h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                               pct(r["after_mover"]) + "%")
            p.setPen(QPen(qc(LINE, .6), 1))
            p.drawLine(QPointF(0, y + self.row_h - .5), QPointF(W, y + self.row_h - .5))

    def _row_at(self, y):
        i = int((y - self.head_h) // self.row_h)
        return i if 0 <= i < len(self.rows) and y >= self.head_h else -1

    def mouseMoveEvent(self, e):
        i = self._row_at(e.position().y())
        if i != self.hover:
            self.hover = i
            self.update()

    def leaveEvent(self, _):
        self.hover = -1
        self.update()

    def mouseReleaseEvent(self, e):
        i = self._row_at(e.position().y())
        if i >= 0 and e.button() == Qt.MouseButton.LeftButton:
            self.on_play(self.rows[i]["uci"])


# ------------------------------------------------------------------ 그래프
class ChartWidget(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setMinimumHeight(150)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def series(self, run):
        out, prev = [], {"w": 1 / 3, "d": 1 / 3, "b": 1 / 3}
        for p in run["positions"]:
            w = wdl_of(p, self.win.S.curve) or prev
            out.append(w)
            prev = w
        return out

    def area(self):
        pl, pr, pt, pb = 40, 10, 8, 24
        return pl, pt, self.width() - pl - pr, self.height() - pt - pb

    def paintEvent(self, _):
        win, S = self.win, self.win.S
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pl, pt, iw, ih = self.area()
        N = max(1, win.n_pos() - 1)
        X = lambda i: pl + iw * i / N
        Y = lambda v: pt + ih * (1 - v)
        has = bool(S.A) and len(S.A["positions"]) > 1
        p.setPen(QPen(qc(INK), 1))
        p.setBrush(qc(B_WIN if has else SURFACE))
        p.drawRect(QRectF(pl, pt, iw, ih))
        if has:
            s = self.series(S.A)

            def poly(f):
                pts = [QPointF(X(0), Y(0))] + [QPointF(X(i), Y(f(w))) for i, w in enumerate(s)] + [QPointF(X(len(s) - 1), Y(0))]
                return QPolygonF(pts)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qc(DRAW))
            p.drawPolygon(poly(lambda w: w["w"] + w["d"]))
            p.setBrush(qc(W_WIN))
            p.drawPolygon(poly(lambda w: w["w"]))
            if S.B:
                sb = self.series(S.B)
                pen = QPen(qc(COBALT), 2.2)
                pen.setDashPattern([3, 2])
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                for f in (lambda w: w["w"], lambda w: w["w"] + w["d"]):
                    p.drawPolyline(QPolygonF([QPointF(X(i), Y(f(w))) for i, w in enumerate(sb)]))
            pen = QPen(qc(INK, .35), 1)
            pen.setDashPattern([2, 4])
            p.setPen(pen)
            p.drawLine(QPointF(pl, Y(.5)), QPointF(pl + iw, Y(.5)))
            d = win.divergence()
            if 0 <= d < N:
                pen = QPen(qc(INK2), 1.5)
                pen.setDashPattern([2, 2])
                p.setPen(pen)
                p.drawLine(QPointF(X(d), pt), QPointF(X(d), pt + ih))
            p.setPen(QPen(qc(MARK), 3))
            p.drawLine(QPointF(X(S.ply), pt - 2), QPointF(X(S.ply), pt + ih + 2))
        else:
            p.setPen(qc(INK2))
            p.setFont(font(SANS, 13))
            p.drawText(QRectF(pl, pt, iw, ih), Qt.AlignmentFlag.AlignCenter,
                       "수를 두거나 기보를 분석하면, 수마다 백 승, 무승부, 흑 승 확률이 쌓여서 그려집니다.")
        p.setFont(font(SANS, 11))
        p.setPen(qc(INK2))
        for v in (0, .5, 1):
            p.drawText(QRectF(0, Y(v) - 8, pl - 6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{int(v * 100)}%")
        if has:
            step = 20 if N > 80 else 10 if N > 30 else 4 if N > 8 else 2
            for i in range(0, N + 1, step):
                p.drawText(QRectF(X(i) - 30, self.height() - 20, 60, 18), Qt.AlignmentFlag.AlignCenter,
                           "시작" if i == 0 else f"{math.ceil(i / 2)}수")

    def seek(self, e):
        S = self.win.S
        if not S.A or len(S.A["positions"]) < 2:
            return
        pl, _, iw, _ = self.area()
        N = max(1, self.win.n_pos() - 1)
        self.win.go(round((e.position().x() - pl) / iw * N))

    def mousePressEvent(self, e):
        self.seek(e)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self.seek(e)


class MoveButton(QPushButton):
    def __init__(self, text, rare=False, var=False):
        super().__init__(text)
        self.rare = rare
        self.setCheckable(True)
        self.setProperty("kind", "move")
        if var:
            self.setFont(font(SERIF, 15, 500, italic=True))

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.rare:
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            fm = QFontMetricsF(self.font())
            x = 8 + fm.horizontalAdvance(self.text()) + 5
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qc(MARK))
            p.drawEllipse(QPointF(min(x, self.width() - 5), self.height() / 2 - 3), 3, 3)


# ------------------------------------------------------------------ 메인 창
class Bridge(QObject):
    done = Signal(object, object)
    failed = Signal(object, str)


def label(text="", role=None, wrap=False, rich=False):
    """rich=True 이면 HTML 로 해석 (값은 호출하는 쪽에서 esc 로 감쌈)."""
    lb = QLabel()
    lb.setTextFormat(Qt.TextFormat.RichText if rich else Qt.TextFormat.PlainText)
    lb.setText(text)
    if role:
        lb.setProperty("role", role)
    lb.setWordWrap(wrap)
    return lb


def esc(v):
    return html.escape(str(v))


def button(text, kind=None):
    b = QPushButton(text)
    if kind:
        b.setProperty("kind", kind)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def clear_layout(lay):
    while lay.count():
        it = lay.takeAt(0)
        w = it.widget()
        if w is not None:
            w.hide()                 # 지우기 전에 바로 숨김 (deleteLater 는 다음 이벤트 때 지워져서 겹쳐 보일 수 있음)
            w.setParent(None)
            w.deleteLater()
        elif it.layout() is not None:
            clear_layout(it.layout())


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.S = State()
        self.an = None
        self.bridge = Bridge()
        self.bridge.done.connect(self._on_done)
        self.bridge.failed.connect(self._on_failed)
        self._jobs = {}
        self._moves_key = None
        self.veil = QTimer(self)
        self.veil.setSingleShot(True)
        self.veil.timeout.connect(self._show_veil)
        self.setWindowTitle("체스 승률 예측")
        self.build()
        self.render_all()
        self.load_model()

    # ---------------------------------------------------------- 화면 구성
    def build(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(26, 18, 26, 18)
        outer.setSpacing(14)

        head = QHBoxLayout()
        head.setSpacing(18)
        t = label("체스 승률 예측", "h1")
        head.addWidget(t, 0, Qt.AlignmentFlag.AlignBottom)
        self.status = label("모델을 불러오는 중입니다.", "status", wrap=True)
        head.addWidget(self.status, 1, Qt.AlignmentFlag.AlignBottom)
        outer.addLayout(head)
        rule = QFrame()
        rule.setFixedHeight(1)
        rule.setStyleSheet(f"background:{LINE}")
        outer.addWidget(rule)

        main = QHBoxLayout()
        main.setSpacing(28)
        outer.addLayout(main, 1)

        # 왼쪽: 입력
        rail_w = QWidget()
        rail = QVBoxLayout(rail_w)
        rail.setContentsMargins(0, 0, 8, 0)
        rail.setSpacing(10)
        rail.addWidget(label("기보 (PGN)", "h2"))
        self.pgn = QPlainTextEdit()
        self.pgn.setPlaceholderText("Lichess나 Chess.com에서 복사한 PGN을 붙여 넣으세요.\n수만 적어도 됩니다. 예: 1. e4 e5 2. Nf3 Nc6")
        self.pgn.setMinimumHeight(170)
        rail.addWidget(self.pgn)
        row = QHBoxLayout()
        self.btn_analyze = button("분석하기", "primary")
        self.btn_sample = button("예시 기보 넣기", "quiet")
        row.addWidget(self.btn_analyze)
        row.addWidget(self.btn_sample)
        row.addStretch(1)
        rail.addLayout(row)
        self.err = label("", "err", wrap=True)
        self.err.hide()
        rail.addWidget(self.err)
        self.notes = QVBoxLayout()
        self.notes.setSpacing(4)
        rail.addLayout(self.notes)

        self.sec_a = QWidget()
        a = QVBoxLayout(self.sec_a)
        a.setContentsMargins(0, 10, 0, 0)
        a.setSpacing(8)
        a.addWidget(self._rule())
        a.addWidget(label("레이팅", "h2"))
        a.addWidget(label("수 예측과 승률 계산에 들어가는 두 사람의 레이팅입니다. PGN을 분석하면 PGN 값으로 채워집니다.", "note", True))
        self.wA, self.bA = self._spin(), self._spin()
        a.addLayout(self._elo_grid(self.wA, self.bA))
        self.btn_applyA = button("이 레이팅으로 다시 계산")
        a.addWidget(self.btn_applyA, alignment=Qt.AlignmentFlag.AlignLeft)
        rail.addWidget(self.sec_a)

        self.sec_b = QWidget()
        b = QVBoxLayout(self.sec_b)
        b.setContentsMargins(0, 10, 0, 0)
        b.setSpacing(8)
        b.addWidget(self._rule())
        b.addWidget(label("다른 레이팅과 비교", "h2"))
        b.addWidget(label("한쪽 레이팅만 바꾸면, 같은 포지션에서 그 레이팅의 사람이 어떤 수를 고를지 나란히 볼 수 있습니다. "
                          "보드에서 수를 두면 비교도 같이 다시 계산됩니다.", "note", True))
        self.wB, self.bB = self._spin(), self._spin()
        b.addLayout(self._elo_grid(self.wB, self.bB))
        row = QHBoxLayout()
        self.btn_compare = button("비교 레이팅으로 계산", "cobalt")
        self.btn_clearB = button("비교 지우기", "quiet")
        row.addWidget(self.btn_compare)
        row.addWidget(self.btn_clearB)
        row.addStretch(1)
        b.addLayout(row)
        b.addWidget(label("승률이 달라지는 몫은 대부분 두 사람의 레이팅 차이에서 옵니다. 레이팅에 따라 수 선택이 달라지는지는 "
                          "오른쪽 표의 고를 확률을 비교해 보세요.", "note", True))
        rail.addWidget(self.sec_b)

        opts = QWidget()
        o = QVBoxLayout(opts)
        o.setContentsMargins(0, 10, 0, 0)
        o.setSpacing(6)
        o.addWidget(self._rule())
        o.addWidget(label("계산 설정", "h3"))
        o.addWidget(label("후보 수를 고르는 누적 확률", "note"))
        self.top_p = QComboBox()
        for text, v in (("80%", 0.8), ("90% (평가 때와 같음)", 0.9), ("95%", 0.95)):
            self.top_p.addItem(text, v)
        self.top_p.setCurrentIndex(1)
        o.addWidget(self.top_p)
        self.mode_label = label("둔 뒤 포지션 평가", "note")
        self.mode = QComboBox()
        self.mode.addItem("모델", "model")
        self.mode.addItem("Stockfish + 보정", "engine")
        o.addWidget(self.mode_label)
        o.addWidget(self.mode)
        self.mode_label.hide()
        self.mode.hide()
        o.addWidget(label("바꾼 설정은 다음 계산부터 적용됩니다.", "note", True))
        rail.addWidget(opts)
        rail.addStretch(1)
        main.addWidget(self._scroll(rail_w, 300))
        main.addStretch(1)

        # 가운데: 보드
        self.board_col = QWidget()
        col = QVBoxLayout(self.board_col)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)
        self.tag_top = label("", "ptag", rich=True)
        self.tag_bot = label("", "ptag", rich=True)
        self.board = BoardWidget(self)
        col.addWidget(self.tag_top)
        col.addWidget(self.board, 1)
        col.addWidget(self.tag_bot)
        nav = self.nav_row = QHBoxLayout()
        self.btn_first, self.btn_prev = button("처음"), button("이전")
        self.btn_next, self.btn_last = button("다음"), button("끝")
        self.where = label("시작 포지션", "where")
        self.where.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for w in (self.btn_first, self.btn_prev):
            nav.addWidget(w)
        nav.addWidget(self.where, 1)
        for w in (self.btn_next, self.btn_last):
            nav.addWidget(w)
        col.addLayout(nav)
        lb = self.line_row = QHBoxLayout()
        self.line_info = label("", "note", True)
        self.btn_back = button("원래 기보로")
        self.btn_new = button("새 판")
        lb.addWidget(self.line_info, 1)
        lb.addWidget(self.btn_back)
        lb.addWidget(self.btn_new)
        col.addLayout(lb)
        bo = self.opts_row = QHBoxLayout()
        self.chk_played = QCheckBox("실제로 둔 수 점선으로 표시")
        self.chk_played.setChecked(True)
        self.chk_flip = QCheckBox("판 뒤집기")
        bo.addWidget(self.chk_played)
        bo.addWidget(self.chk_flip)
        bo.addStretch(1)
        self.arrow_seg = QWidget()
        sl = QHBoxLayout(self.arrow_seg)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)
        sl.addWidget(label("화살표 ", "note"))
        self.seg_A, self.seg_B = button("기준", "segcob"), button("비교", "segcob")
        self.seg_A.setToolTip("기준 레이팅의 후보 수를 화살표로")
        self.seg_B.setToolTip("비교 레이팅의 후보 수를 화살표로")
        for s in (self.seg_A, self.seg_B):
            s.setCheckable(True)
            sl.addWidget(s)
        bo.addWidget(self.arrow_seg)
        col.addLayout(bo)
        main.addWidget(self.board_col, 10)
        main.addStretch(1)

        # 오른쪽: 포지션 분석
        side_w = QWidget()
        self.side = QVBoxLayout(side_w)
        self.side.setContentsMargins(0, 0, 8, 0)
        self.side.setSpacing(12)
        main.addWidget(self._scroll(side_w, 390))

        # 아래: 그래프 + 기보
        bottom = QHBoxLayout()
        bottom.setSpacing(28)
        chart_box = QVBoxLayout()
        ch = QHBoxLayout()
        ch.addWidget(label("대국 전체 흐름", "h2"))
        ch.addStretch(1)
        self.seg_w, self.seg_d = button("수별 가중", "seg"), button("직접 예측", "seg")
        for s in (self.seg_w, self.seg_d):
            s.setCheckable(True)
            ch.addWidget(s)
        chart_box.addLayout(ch)
        self.chart = ChartWidget(self)
        chart_box.addWidget(self.chart, 1)
        self.legend = label("", "note", rich=True)
        chart_box.addWidget(self.legend)
        bottom.addLayout(chart_box, 3)
        moves_box = QVBoxLayout()
        moves_box.addWidget(label("기보", "h2"))
        self.moves_w = QWidget()
        self.moves_box = QVBoxLayout(self.moves_w)
        self.moves_box.setContentsMargins(0, 0, 0, 0)
        self.moves_scroll = self._scroll(self.moves_w, None)
        moves_box.addWidget(self.moves_scroll, 1)
        self.moves_note = label("", "note", rich=True)
        moves_box.addWidget(self.moves_note)
        bottom.addLayout(moves_box, 2)
        bw = QWidget()
        bw.setLayout(bottom)
        bw.setFixedHeight(280)
        outer.addWidget(bw)

        # 연결
        self.btn_analyze.clicked.connect(self.analyze_pgn)
        self.btn_sample.clicked.connect(lambda: (self.pgn.setPlainText(SAMPLE), self.pgn.setFocus()))
        self.btn_applyA.clicked.connect(self.apply_a)
        self.btn_compare.clicked.connect(self.compare_b)
        self.btn_clearB.clicked.connect(self.clear_b)
        self.btn_new.clicked.connect(self.new_board)
        self.btn_back.clicked.connect(self.back_orig)
        self.btn_first.clicked.connect(lambda: self.go(0))
        self.btn_prev.clicked.connect(lambda: self.go(self.S.ply - 1))
        self.btn_next.clicked.connect(lambda: self.go(self.S.ply + 1))
        self.btn_last.clicked.connect(lambda: self.go(self.n_pos() - 1))
        self.chk_flip.toggled.connect(lambda v: (setattr(self.S, "flip", v), self.render_all()))
        self.chk_played.toggled.connect(lambda v: (setattr(self.S, "show_played", v), self.board.update()))
        self.seg_A.clicked.connect(lambda: self.set_arrows("A"))
        self.seg_B.clicked.connect(lambda: self.set_arrows("B"))
        self.seg_w.clicked.connect(lambda: self.set_curve("weighted"))
        self.seg_d.clicked.connect(lambda: self.set_curve("direct"))
        QShortcut(QKeySequence("Ctrl+Return"), self.pgn, activated=self.analyze_pgn, context=Qt.ShortcutContext.WidgetShortcut)
        self.wB.returnPressed.connect(self.compare_b)
        self.bB.returnPressed.connect(self.compare_b)
        self.wA.returnPressed.connect(self.apply_a)
        self.bA.returnPressed.connect(self.apply_a)

    def keyPressEvent(self, e):
        """← → Home End 로 이동, Esc 로 선택 해제. 글 입력칸이 키를 먼저 쓰고, 안 쓴 키만 여기로 옴."""
        k, S = e.key(), self.S
        if k == Qt.Key.Key_Left:
            self.go(S.ply - 1)
        elif k == Qt.Key.Key_Right:
            self.go(S.ply + 1)
        elif k == Qt.Key.Key_Home:
            self.go(0)
        elif k == Qt.Key.Key_End:
            self.go(self.n_pos() - 1)
        elif k == Qt.Key.Key_Escape:
            self.deselect()
        else:
            super().keyPressEvent(e)

    def _rule(self):
        r = QFrame()
        r.setFixedHeight(1)
        r.setStyleSheet(f"background:{LINE}")
        return r

    def _spin(self):
        # QSpinBox 는 글꼴을 스타일로 바꾸면 숫자가 안 보이는 경우가 있어 숫자 전용 QLineEdit 사용
        s = QLineEdit()
        s.setValidator(QIntValidator(0, 9999, s))
        s.setProperty("role", "elo")
        s.setMaxLength(4)
        return s

    def _elo_grid(self, w, b):
        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(3)
        g.addWidget(label("백", "note"), 0, 0)
        g.addWidget(label("흑", "note"), 0, 1)
        g.addWidget(w, 1, 0)
        g.addWidget(b, 1, 1)
        return g

    def _scroll(self, inner, width):
        sc = QScrollArea()
        sc.setWidget(inner)
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.Shape.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        if width:
            sc.setFixedWidth(width)
        return sc

    # ---------------------------------------------------------- 상태 도우미
    def cur_pos(self, run):
        if not run:
            return None
        return run["positions"][min(self.S.ply, len(run["positions"]) - 1)]

    def n_pos(self):
        return len(self.S.A["positions"]) if self.S.A else 1

    def line_uci(self):
        return [p["last_move"] for p in self.S.A["positions"][1:]] if self.S.A else []

    def divergence(self):
        S = self.S
        if not S.orig:
            return -1
        cur, o = self.line_uci(), S.orig["moves"]
        i = 0
        while i < len(cur) and i < len(o) and cur[i] == o[i]:
            i += 1
        return i

    def can_play(self):
        p = self.cur_pos(self.S.A)
        return bool(p) and not self.S.busy and bool(p.get("legal"))

    # ---------------------------------------------------------- 계산 (별도 스레드)
    def run_async(self, fn, ok, fail):
        token = object()
        self._jobs[token] = (ok, fail)

        def target():
            try:
                r = fn()
            except SystemExit as e:          # resolve_ckpt 의 "체크포인트 없음" 등
                self.bridge.failed.emit(token, str(e))
                return
            except Exception as e:
                from analysis import UserError
                if isinstance(e, UserError):
                    self.bridge.failed.emit(token, str(e))
                else:
                    traceback.print_exc()
                    self.bridge.failed.emit(token, f"계산 중 오류가 났습니다: {type(e).__name__}: {e}")
                return
            self.bridge.done.emit(token, r)
        threading.Thread(target=target, daemon=True).start()

    def _on_done(self, token, r):
        ok, _ = self._jobs.pop(token)
        ok(r)

    def _on_failed(self, token, msg):
        _, fail = self._jobs.pop(token)
        fail(msg)

    def load_model(self):
        import torch
        from analysis import Analyzer, resolve_ckpt
        a = self.args

        def job():
            device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            ckpt, latest = resolve_ckpt(a.ckpt)
            return Analyzer(ckpt, device, a.engine, a.calib, a.depth, a.threads, a.hash, a.top_p, latest)

        def ok(an):
            self.an = an
            i = an.info()
            months = f", {len(i['months'])}개월치 대국으로 학습 ({', '.join(i['months'])})" if i.get("months") else ""
            dev = "GPU" if i["device"].startswith("cuda") else "CPU"
            eng = f", 엔진 평가 사용 ({i['engine']})" if i.get("engine") else ""
            self.status.setText(f"현재 모델 {i['ckpt'].split('/')[-1]}{months}. {dev}에서 실행 중{eng}.")
            if "engine" in i["weighted_modes"]:
                self.mode_label.show()
                self.mode.show()
            self.new_board()

        def fail(msg):
            self.status.setText("모델을 불러오지 못했습니다: " + msg)
            self.status.setProperty("role", "statusbad")
            self.status.style().unpolish(self.status)
            self.status.style().polish(self.status)
        self.set_busy(True)
        self.run_async(job, lambda an: (self.set_busy(False), ok(an)), lambda m: (self.set_busy(False), fail(m)))

    def request(self, line, elo):
        base = {"pgn": line["pgn"]} if line["kind"] == "pgn" else \
            {"moves": line["moves"], "white": line.get("white"), "black": line.get("black"), "time_control": line.get("tc")}
        base.update({"top_p": self.top_p.currentData(), "weighted": self.mode.currentData(),
                     "white_elo": elo["w"] if elo else None, "black_elo": elo["b"] if elo else None})
        return base

    def compute(self, line, eloA, eloB, ply, then=None):
        if self.S.busy or not self.an:
            return
        self.set_error("")
        self.set_busy(True)
        an = self.an

        def job():
            A = an.analyze(self.request(line, eloA))
            B = an.analyze(self.request(line, eloB)) if eloB else None
            return A, B

        def ok(res):
            S = self.S
            S.A, S.B = res
            S.line, S.eloB = line, eloB
            S.eloA = eloA or {"w": S.A["meta"]["white_elo"], "b": S.A["meta"]["black_elo"]}
            S.ply = max(0, min(ply, len(S.A["positions"]) - 1))
            S.sel, S.drag, S.show_all = None, None, False
            if then:
                then()
            self.set_busy(False)
            self.sync_fields()
            self.render_notes()
            self.render_all()

        def fail(msg):
            self.set_busy(False)
            self.set_error(msg)
            self.render_all()
        self.run_async(job, ok, fail)

    def set_busy(self, on):
        self.S.busy = on
        for b in (self.btn_analyze, self.btn_compare, self.btn_applyA, self.btn_new, self.btn_back):
            b.setEnabled(not on)
        self.btn_analyze.setText("분석 중" if on and self.S.line and self.S.line.get("kind") == "pgn" else "분석하기")
        if on:
            self.veil.start(220)
        else:
            self.veil.stop()
            self.board.busy_text = False
            self.board.update()

    def _show_veil(self):
        self.board.busy_text = True
        self.board.update()

    def set_error(self, msg):
        self.err.setText(msg)
        self.err.setVisible(bool(msg))

    # ---------------------------------------------------------- 동작
    def read_elo(self, w, b):
        try:
            e = {"w": int(w.text()), "b": int(b.text())}
        except ValueError:
            e = None
        if not e or not all(100 <= v <= 3449 for v in e.values()):
            self.set_error("레이팅은 100~3449 사이의 숫자로 입력하세요.")
            return None
        return e

    def analyze_pgn(self):
        text = self.pgn.toPlainText()
        if not text.strip():
            self.set_error("PGN이 비어 있습니다. 기보를 붙여 넣거나 예시 기보를 넣어 보세요.")
            return

        def then():
            self.S.orig = None
            self.S.arrows = "B"
        self.compute({"kind": "pgn", "pgn": text}, None, None, 0, then)

    def apply_a(self):
        e = self.read_elo(self.wA, self.bA) if self.S.A else None
        if e:
            self.compute(self.S.line, e, self.S.eloB, self.S.ply)

    def compare_b(self):
        S = self.S
        if not S.A:
            return
        e = self.read_elo(self.wB, self.bB)
        if not e:
            return
        if e == S.eloA:
            self.set_error("기준 레이팅과 다른 값을 하나 이상 넣은 뒤 계산하세요.")
            return
        S.arrows = "B"
        self.compute(S.line, S.eloA, e, S.ply)

    def clear_b(self):
        self.S.eloB = None
        self.S.B = None
        self.sync_fields()
        self.render_notes()
        self.render_all()

    def new_board(self):
        self.compute({"kind": "moves", "moves": [], "white": None, "black": None, "tc": None}, self.S.eloA, self.S.eloB, 0,
                     lambda: setattr(self.S, "orig", None))

    def back_orig(self):
        o = self.S.orig
        if o:
            self.compute(o["line"], self.S.eloA, self.S.eloB, o["ply"], lambda: setattr(self.S, "orig", None))

    def play_move(self, uci):
        S = self.S
        if not S.A or S.busy:
            return
        cur = self.line_uci()
        if S.ply < len(cur) and cur[S.ply] == uci:      # 기보와 같은 수 = 다음 포지션으로 이동
            self.go(S.ply + 1)
            return
        m = S.A["meta"]
        line = {"kind": "moves", "moves": cur[:S.ply] + [uci], "white": m["white"], "black": m["black"], "tc": m["time_control"]}
        orig = S.orig or ({"line": S.line, "ply": S.ply, "moves": cur} if S.line["kind"] == "pgn" else None)
        self.compute(line, S.eloA, S.eloB, S.ply + 1, lambda: setattr(S, "orig", orig))

    def try_move(self, frm, to):
        S = self.S
        opts = [u for t, u in legal_map(self.cur_pos(S.A)).get(frm, []) if t == to]
        S.sel = None
        if not opts:
            self.board.update()
            return
        if len(opts) > 1:
            dlg = PromoDialog(self, self.cur_pos(S.A)["white_to_move"], opts)
            self.board.update()
            if dlg.exec() and dlg.choice:
                self.play_move(dlg.choice)
            return
        self.play_move(opts[0])

    def go(self, ply):
        S = self.S
        S.ply = max(0, min(self.n_pos() - 1, ply))
        S.show_all, S.sel = False, None
        self.render_all()

    def deselect(self):
        if self.S.sel is not None:
            self.S.sel = None
            self.board.update()

    def set_arrows(self, run):
        self.S.arrows = run
        self.render_nav()
        self.board.update()

    def set_curve(self, c):
        self.S.curve = c
        self.render_chart_head()
        self.chart.update()

    # ---------------------------------------------------------- 그리기
    def sync_fields(self):
        S = self.S
        has = bool(S.A)
        self.sec_a.setVisible(has)
        self.sec_b.setVisible(has)
        if not has:
            return
        self.wA.setText(str(S.eloA["w"]))
        self.bA.setText(str(S.eloA["b"]))
        B = S.eloB or S.eloA
        self.wB.setText(str(B["w"]))
        self.bB.setText(str(B["b"]))
        for sp, changed in ((self.wB, bool(S.eloB) and S.eloB["w"] != S.eloA["w"]),
                            (self.bB, bool(S.eloB) and S.eloB["b"] != S.eloA["b"])):
            sp.setProperty("changed", changed)
            sp.style().unpolish(sp)
            sp.style().polish(sp)
        self.btn_clearB.setVisible(bool(S.eloB))

    def render_notes(self):
        clear_layout(self.notes)
        S = self.S
        if not S.A:
            return
        m = S.A["meta"]
        add = lambda t: self.notes.addWidget(label(t, "note", True, rich=True))
        if S.line["kind"] == "pgn":
            missing = [n for c, n in (("white", "백"), ("black", "흑")) if not m["elo_in_pgn"][c]]
            if missing:
                add(f"PGN에 {'과 '.join(missing)} 레이팅이 없어 1500으로 계산했습니다. 아래 레이팅 칸에서 바꿀 수 있습니다.")
            if m["games_in_pgn"] > 1:
                add(f"PGN에 대국이 {m['games_in_pgn']}판 있어 첫 판만 분석했습니다.")
            if not m["has_clock"]:
                add("시계 기록(%clk)이 없어 남은 시간 정보 없이 계산했습니다.")
        else:
            add("보드에서 말을 끌어 놓거나, 말을 누른 뒤 갈 칸을 누르면 둡니다. 오른쪽 표의 수를 눌러도 됩니다.")
            if S.orig:
                add("직접 둔 수순은 시계 정보 없이 계산하므로, 갈라지기 전 포지션의 숫자도 기보 분석 때와 조금 다를 수 있습니다.")
        known = [m[c] for c in ("white", "black") if m["known_players"][c]]
        if known:
            add(f"<b>{esc(', '.join(known))}</b>은(는) 학습 데이터에 자주 나온 플레이어라 개인 특성이 반영됩니다.")
        cached = f", 이전 계산 재사용 {m['cached_positions']}개" if m.get("cached_positions") else ""
        extra = f" 비교 계산 {S.B['meta']['seconds']}초." if S.B else ""
        add(f"계산 {m['seconds']}초 ({m['n_plies']}수{cached}).{extra}")

    def render_all(self):
        self.render_players()
        self.board.update()
        self.render_nav()
        self.render_side()
        self.render_chart_head()
        self.chart.update()
        self.render_moves()

    def render_players(self):
        S = self.S
        m = S.A["meta"] if S.A else None
        mb = S.B["meta"] if S.B else None

        def tag(white):
            c = "white" if white else "black"
            dot = f"<span style='color:{INK}'>{'○' if white else '●'}</span>"
            name = m[c] if m and m[c] and m[c] != "?" else side(white)
            s = f"{dot}&nbsp;&nbsp;<b>{esc(name)}</b>"
            if m:
                s += f"&nbsp;&nbsp;<span style=\"font-family:'{SERIF}';font-weight:600\">{m[c + '_elo']}</span>"
            if mb and mb[c + "_elo"] != m[c + "_elo"]:
                s += f"&nbsp;&nbsp;<span style='color:{COBALT}'>비교 {mb[c + '_elo']}</span>"
            return s
        self.tag_top.setText(tag(S.flip))
        self.tag_bot.setText(tag(not S.flip))

    def render_nav(self):
        S = self.S
        pos = self.cur_pos(S.A)
        self.where.setText("시작 포지션" if not pos or S.ply == 0 else move_label(S.ply - 1, S.A["positions"][S.ply - 1]["next_san"]))
        self.btn_first.setEnabled(S.ply > 0)
        self.btn_prev.setEnabled(S.ply > 0)
        self.btn_next.setEnabled(S.ply < self.n_pos() - 1)
        self.btn_last.setEnabled(S.ply < self.n_pos() - 1)
        self.arrow_seg.setVisible(bool(S.B))
        self.seg_A.setChecked(S.arrows == "A")
        self.seg_B.setChecked(S.arrows == "B")
        info = ""
        if S.A:
            n, d = S.A["meta"]["n_plies"], self.divergence()
            if S.line["kind"] == "pgn":
                info = f"붙여 넣은 기보 ({n}수)"
            elif S.orig:
                if d >= n:
                    info = "기보의 앞부분까지만 남긴 수순"
                elif d >= len(S.orig["moves"]):
                    info = f"기보 끝에 이어서 둔 수순: {move_label(d, S.A['positions'][d]['next_san'])}부터"
                else:
                    info = f"기보와 다른 수순: {move_label(d, S.A['positions'][d]['next_san'])}부터 직접 둠"
            else:
                info = "새 판입니다. 말을 움직여 보세요." if n == 0 else f"직접 둔 수순 ({n}수)"
        self.line_info.setText(info)
        self.btn_back.setVisible(bool(S.orig))

    def render_chart_head(self):
        self.seg_w.setChecked(self.S.curve == "weighted")
        self.seg_d.setChecked(self.S.curve == "direct")
        sw = lambda c, border=INK: (f"<span style='background:{c};color:{c};border:1px solid {border}'>&nbsp;&nbsp;&nbsp;</span>")
        parts = [f"{sw(W_WIN)} 백 승", f"{sw(DRAW)} 무승부", f"{sw(B_WIN)} 흑 승", f"<span style='color:{MARK}'>┃</span> 지금 보는 포지션"]
        if self.S.B:
            parts.append(f"<span style='color:{COBALT}'>╌╌</span> 비교 레이팅의 경계")
        self.legend.setText("&nbsp;&nbsp;&nbsp;&nbsp;".join(parts))

    def render_side(self):
        clear_layout(self.side)
        S = self.S
        pos = self.cur_pos(S.A)
        add = self.side.addWidget
        if not pos:
            head = QHBoxLayout()
            head.addWidget(label("포지션 분석", "h2"))
            self.side.addLayout(head)
            e = label("모델을 불러오면 시작 포지션 분석이 여기에 나옵니다. 보드에서 말을 직접 두거나, 왼쪽에 기보를 붙여 넣어 분석할 수 있습니다.",
                      "empty", True)
            add(e)
            self.side.addStretch(1)
            return
        title = "시작 포지션" if S.ply == 0 else f"{move_label(S.ply - 1, S.A['positions'][S.ply - 1]['next_san'])} 둔 뒤"
        head = QHBoxLayout()
        head.addWidget(label(title, "h2"))
        head.addStretch(1)
        head.addWidget(label(f"{side(pos['white_to_move'])} 차례", "note"))
        self.side.addLayout(head)
        add(self._rule())
        if not pos.get("model"):
            msg = (f"체크메이트로 {side(not pos['white_to_move'])}이 이겼습니다." if pos["terminal"] == "checkmate"
                   else TERM.get(pos["terminal"], "대국이 끝난 포지션입니다."))
            add(label(msg, "empty", True))
            self.side.addStretch(1)
            return
        posB = self.cur_pos(S.B)
        if S.B:
            self.win_compare(pos, posB)
            self.cand_compare(pos, posB)
        else:
            self.win_single(pos)
            self.cand_single(pos)
        self.played_line(pos, posB)
        self.side.addStretch(1)

    def _block_title(self, title, sub):
        row = QHBoxLayout()
        row.addWidget(label(title, "h2"))
        row.addStretch(1)
        s = label(sub, "note", True)
        s.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(s, 1)
        self.side.addLayout(row)

    def win_single(self, pos):
        m = pos["model"]
        wt = m.get("weighted")
        self._block_title("승률", "사람이 둘 법한 수를 하나씩 두어 보고, 고를 확률로 평균한 값" if wt else "직접 예측")
        self.side.addWidget(HBar(wt["conditional"] if wt else m["direct"]))
        if wt:
            r = wt["range"]
            e = lambda v: esc(pct(v))
            self.side.addWidget(label(f"후보에 들지 않은 수({e(wt['excluded_mass'])}%)까지 생각하면 백 승률은 "
                                      f"<b>{e(r['w'][0])}~{e(r['w'][1])}%</b>, 흑 승률은 <b>{e(r['b'][0])}~{e(r['b'][1])}%</b>입니다.",
                                      "para", True, rich=True))
            d = m["direct"]
            self.side.addWidget(label(f"직접 예측으로는 <b>백 {e(d['w'])}%, 무 {e(d['d'])}%, 흑 {e(d['b'])}%</b>입니다.",
                                      "para", True, rich=True))
        if m.get("eval"):
            self.side.addWidget(label(f"엔진 평가 {eng_txt(m['eval'])} (백 기준)", "para", True))

    def win_compare(self, pos, posB):
        mA, mB = self.S.A["meta"], self.S.B["meta"]
        wA, wB = wdl_of(pos), wdl_of(posB)
        self._block_title("승률", "수별 가중" if pos["model"].get("weighted") else "직접 예측")
        self.side.addWidget(label(f"<b>기준 레이팅</b>&nbsp; 백 {mA['white_elo']}, 흑 {mA['black_elo']}", "note", rich=True))
        self.side.addWidget(HBar(wA, slim=True))
        self.side.addWidget(label(f"<b style='color:{COBALT}'>비교 레이팅</b>&nbsp; 백 {mB['white_elo']}, 흑 {mB['black_elo']}", "note", rich=True))
        self.side.addWidget(HBar(wB, slim=True, cobalt=True))
        d = round((wB["w"] - wA["w"]) * 100)
        self.side.addWidget(label("백 승률은 거의 같습니다." if d == 0 else
                                  f"비교 레이팅에서 백 승률이 {abs(d)}%p {'높습니다' if d > 0 else '낮습니다'}.", "para", True))

    def _more_button(self, total):
        if total <= 10:
            return
        b = button("10개만 보기" if self.S.show_all else f"나머지 {total - 10}개 보기", "quiet")
        b.clicked.connect(lambda: (setattr(self.S, "show_all", not self.S.show_all), self.render_side()))
        self.side.addWidget(b, alignment=Qt.AlignmentFlag.AlignLeft)

    def cand_single(self, pos):
        m = pos["model"]
        wt = m.get("weighted")
        mover = side(pos["white_to_move"])
        lst = wt["candidates"] if wt else m["policy"]
        sub = (f"합법수 {wt['n_legal']}개 중 {len(lst)}개, 고를 확률 합 {pct(wt['selected_mass'])}%" if wt
               else f"확률 상위 {len(lst)}개")
        self._block_title(f"{mover}이 둘 수", sub)
        rows = []
        for c in (lst if self.S.show_all else lst[:10]):
            after = (c["after"]["w"] if pos["white_to_move"] else c["after"]["b"]) if wt else None
            rows.append({"san": c["san"], "uci": c["uci"], "p": c["p"], "after_mover": after, "engine": c.get("engine")})
        has_eng = any(c.get("engine") for c in lst)
        played = m["played"]["san"] if m.get("played") else None
        self.side.addWidget(CandList(rows, "single", self.play_move, mover, bool(wt), has_eng, played))
        self._more_button(len(lst))

    def cand_compare(self, pos, posB):
        mover = side(pos["white_to_move"])
        pa = {c["uci"]: c for c in pos["model"]["policy"]}
        pb = {c["uci"]: c for c in posB["model"]["policy"]}
        ucis = list(dict.fromkeys(list(pa) + list(pb)))
        rows = [{"uci": u, "san": (pa.get(u) or pb.get(u))["san"], "a": pa[u]["p"] if u in pa else None,
                 "b": pb[u]["p"] if u in pb else None} for u in ucis]
        rows.sort(key=lambda r: -max(r["a"] or 0, r["b"] or 0))
        self._block_title(f"{mover}이 둘 수", "고를 확률, 레이팅별 상위 12개")
        played = pos["model"]["played"]["san"] if pos["model"].get("played") else None
        self.side.addWidget(CandList(rows[:24 if self.S.show_all else 10], "compare", self.play_move, mover, played=played))
        self._more_button(len(rows))

    def played_line(self, pos, posB):
        p = pos["model"].get("played")
        if not p:
            self.side.addWidget(label("수순의 마지막 포지션입니다. 보드에서 다음 수를 두거나 표의 수를 눌러 이어 갈 수 있습니다.", "para", True))
            return
        t = (f"다음에 실제로 둔 수는 <b>{esc(p['san'])}</b>입니다. 모델은 이 수를 {esc(pct1(p['p']))}%로 보고 "
             f"합법수 {p['n_legal']}개 중 {p['rank']}순위에 두었습니다.")
        if posB and posB.get("model") and posB["model"].get("played"):
            q = posB["model"]["played"]
            t += f" 비교 레이팅에서는 {esc(pct1(q['p']))}%, {q['rank']}순위입니다."
        self.side.addWidget(label(t, "para", True, rich=True))

    def render_moves(self):
        S = self.S
        n = S.A["meta"]["n_plies"] if S.A else 0
        d = self.divergence()
        key = (id(S.A), d)
        if key != self._moves_key:
            self._moves_key = key
            clear_layout(self.moves_box)
            self._move_btns = []
            gw = QWidget()
            self.moves = QGridLayout(gw)
            self.moves.setContentsMargins(0, 0, 0, 0)
            self.moves.setHorizontalSpacing(2)
            self.moves.setVerticalSpacing(0)
            self.moves_box.addWidget(gw)
            self.moves_box.addStretch(1)
            if not n:
                e = label("아직 둔 수가 없습니다.", "note")
                self.moves.addWidget(e, 0, 0, 1, 3)
            P = S.A["positions"] if S.A else []
            for t in range(0, n, 2):
                r = t // 2
                no = label(f"{r + 1}.", "moveno")
                no.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                no.setFixedWidth(34)
                self.moves.addWidget(no, r, 0)
                for j, k in enumerate((t, t + 1)):
                    if k >= n:
                        continue
                    pd = (P[k].get("model") or {}).get("played")
                    b = MoveButton(P[k]["next_san"], rare=bool(pd and pd["p"] < 0.05), var=d >= 0 and k >= d)
                    if pd:
                        b.setToolTip(f"모델 확률 {pct1(pd['p'])}%, {pd['rank']}순위")
                    b.clicked.connect(lambda _=False, k=k: self.go(k + 1))
                    self.moves.addWidget(b, r, j + 1)
                    self._move_btns.append((k, b))
            self.moves.setColumnStretch(1, 1)
            self.moves.setColumnStretch(2, 1)
            notes = [f"<span style='color:{MARK}'>●</span> 모델이 5% 미만으로 본 수"] if n else []
            if 0 <= d < n:
                notes.append("<i>기울어진 수</i>는 기보와 다르게 직접 둔 수")
            self.moves_note.setText(",&nbsp;&nbsp;".join(notes))
        cur = None
        for k, b in getattr(self, "_move_btns", []):
            on = k + 1 == S.ply
            b.setChecked(on)
            if on:
                cur = b
        if cur:
            self.moves_scroll.ensureWidgetVisible(cur, 0, 20)


STYLE = f"""
QWidget {{ background: {PAPER}; color: {INK}; font-family: "{SANS}"; font-size: 14px; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: {PAPER}; }}
QLabel[role="h1"] {{ font-family: "{SERIF}"; font-size: 27px; font-weight: 600; }}
QLabel[role="h2"] {{ font-family: "{SERIF}"; font-size: 17px; font-weight: 600; }}
QLabel[role="h3"] {{ font-size: 13px; font-weight: 700; color: {INK2}; }}
QLabel[role="status"] {{ color: {INK2}; font-size: 13px; }}
QLabel[role="statusbad"] {{ color: {ERROR}; font-size: 13px; }}
QLabel[role="note"] {{ color: {INK2}; font-size: 13px; }}
QLabel[role="para"] {{ color: {INK2}; font-size: 13px; }}
QLabel[role="err"] {{ color: {ERROR}; font-size: 13px; }}
QLabel[role="where"] {{ font-family: "{SERIF}"; font-size: 16px; font-weight: 600; }}
QLabel[role="ptag"] {{ font-size: 14px; padding-left: 34px; }}
QLabel[role="empty"] {{ background: {SURFACE}; border: 1px dashed {LINE}; border-radius: 4px; padding: 16px; color: {INK2}; }}
QLabel[role="moveno"] {{ color: {INK2}; font-size: 12px; padding-right: 6px; }}
QPlainTextEdit, QLineEdit, QComboBox {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 4px; padding: 6px 8px; }}
QPlainTextEdit {{ font-size: 13px; }}
QPlainTextEdit:focus, QLineEdit:focus, QComboBox:focus {{ border: 1px solid {COBALT}; }}
QLineEdit[role="elo"] {{ font-family: "{SERIF}"; font-size: 18px; font-weight: 600; padding: 5px 10px; }}
QLineEdit[changed="true"] {{ border: 1px solid {COBALT}; border-left: 4px solid {COBALT}; }}
QComboBox QAbstractItemView {{ background: {SURFACE}; selection-background-color: {INK}; selection-color: {W_WIN}; }}
QPushButton {{ border: 1px solid {INK}; border-radius: 4px; padding: 7px 13px; background: transparent; font-weight: 500; }}
QPushButton:hover {{ background: rgba(28,35,38,0.07); }}
QPushButton:focus {{ border: 2px solid {COBALT}; padding: 6px 12px; }}
QPushButton:disabled {{ color: #8C9598; border-color: #AEB6BA; }}
QPushButton[kind="primary"] {{ background: {INK}; color: {W_WIN}; }}
QPushButton[kind="primary"]:hover {{ background: #2E383C; }}
QPushButton[kind="primary"]:disabled {{ background: #5E686C; color: {W_WIN}; }}
QPushButton[kind="cobalt"] {{ background: {COBALT}; border-color: {COBALT}; color: white; }}
QPushButton[kind="cobalt"]:hover {{ background: #2840A6; }}
QPushButton[kind="quiet"] {{ border: none; color: {INK2}; text-decoration: underline; padding: 7px 4px; }}
QPushButton[kind="seg"], QPushButton[kind="segcob"] {{ border-radius: 0; padding: 4px 10px; font-size: 12px; }}
QPushButton[kind="seg"]:checked {{ background: {INK}; color: {W_WIN}; }}
QPushButton[kind="segcob"]:checked {{ background: {COBALT}; border-color: {COBALT}; color: white; }}
QPushButton[kind="move"] {{ border: none; border-radius: 3px; text-align: left; padding: 3px 8px;
    font-family: "{SERIF}"; font-size: 15px; font-weight: 500; }}
QPushButton[kind="move"]:checked {{ background: {INK}; color: {W_WIN}; }}
QCheckBox {{ color: {INK2}; font-size: 13px; spacing: 6px; }}
QToolTip {{ background: {INK}; color: {W_WIN}; border: none; padding: 4px 6px; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""


def parse_args(argv):
    ap = argparse.ArgumentParser(description="체스 승률 예측 데스크톱 앱")
    ap.add_argument("--ckpt", default=None, help="기본: ckpt/LATEST.json 의 현재 모델")
    ap.add_argument("--device", default=None, help="cuda / cpu (기본: 자동)")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--engine", default=None, help="Stockfish 실행 파일 (모든 포지션의 엔진 평가를 모델 입력에 넣음)")
    ap.add_argument("--calib", default=None, help="엔진 평가 → 승무패 보정 파일 (수별 가중 'Stockfish + 보정' 방식에 필요)")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=64)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    load_fonts()
    app.setFont(font(SANS, 14))
    app.setStyleSheet(STYLE)
    win = MainWindow(args)
    scr = app.primaryScreen().availableGeometry()
    win.resize(min(1480, scr.width() - 40), min(980, scr.height() - 40))
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
