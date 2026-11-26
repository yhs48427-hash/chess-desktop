# 체스 승률 예측 (데스크톱 앱)

[Releases](https://github.com/yhs48427-hash/chess-desktop/releases/latest)에서 **chess_desktop.zip**을 받아 압축을 푼 뒤 **`run.bat`을 더블클릭**하면 창이 하나 뜹니다. Windows 기준입니다. 서버나 브라우저 없이 앱이 모델을 직접 부릅니다.
창이 뜨면 새 판이 준비되어 있어서 바로 말을 움직일 수 있고, PGN을 붙여 넣어 대국 전체를 분석할 수도 있습니다.

## 처음 실행할 때 일어나는 일

1. **실행 환경**: PyTorch가 이미 깔린 Python을 찾습니다. 예전 `chess_app` 폴더의 `.venv`가 있으면 그걸 다시 쓰고,
   화면 라이브러리(PySide6)만 이 폴더의 `_deps`에 따로 설치합니다. 예전 환경 자체는 건드리지 않습니다.
   찾지 못하면 이 폴더에 `.venv`를 만들고 PyTorch(GPU가 있으면 CUDA 버전, 약 2.5GB)부터 모두 설치합니다.
2. **모델 파일**: Release의 chess_desktop.zip에 학습된 모델 `ckpt/round04_2025-04.pt`(35MB)가 들어 있습니다.
   저장소 코드만 받았다면(Code → Download ZIP, `git clone`) Release에서 이 파일을 받아 `ckpt/` 폴더에 넣으세요.
3. 앱 창을 엽니다.

두 번째부터는 3번만 합니다. 검은 창(로그)을 닫으면 앱도 꺼집니다.

## 조작

| 하고 싶은 것 | 방법 |
|---|---|
| 수 두기 | 말을 끌어다 놓기, 또는 말을 누른 뒤 점 표시된 칸 누르기. 오른쪽 후보 표의 행을 눌러도 둠 |
| 승격 | 끝 줄에 두면 기물 고르는 창이 뜸 |
| 선택 해제 | 같은 말을 다시 누르거나 Esc |
| 이동 | ← → Home End, 처음/이전/다음/끝 버튼, 그래프나 기보 목록 누르기 |
| PGN 분석 | 왼쪽 칸에 붙여 넣고 "분석하기" (Ctrl+Enter) |
| 기보에서 갈라지기 | 분석한 기보의 아무 포지션에서 다른 수를 두면 새 수순이 됨. "원래 기보로"로 돌아감 |
| 레이팅 | "레이팅" 칸에 두 사람 레이팅을 넣고 "이 레이팅으로 다시 계산" |
| 레이팅 비교 | "다른 레이팅과 비교"에서 계산하면 파란색으로 나란히 보임. 이후 수를 둘 때마다 같이 다시 계산됨 |

보드의 노란 화살표는 사람이 그 수를 고를 확률(굵을수록 높음), 검은 점선은 기보에서 실제로 둔 다음 수입니다.
보드 왼쪽 세로 막대는 지금 포지션의 백 승 / 무 / 흑 승 확률입니다.

## 옵션

`run.bat` 뒤에 붙이면 앱으로 넘어갑니다.

| 예 | 설명 |
|---|---|
| `run.bat --device cpu` | GPU가 있어도 CPU로 실행 |
| `run.bat --ckpt ckpt/round03_2025-03.pt` | 다른 체크포인트 (파일을 `ckpt/`에 직접 넣어 두세요) |
| `run.bat --engine C:\stockfish\stockfish.exe --calib results/calib.json` | Stockfish 평가를 모델 입력에 넣고, 계산 설정에서 "Stockfish + 보정"을 쓸 수 있게 함 |

새 라운드 모델이 생기면 Drive의 `ckpt/LATEST.json`과 새 `.pt`를 이 폴더의 `ckpt/`에 덮어쓰면 됩니다.

## 폴더 구성

```
chess_desktop/
├─ run.bat            더블클릭으로 실행 (Windows)
├─ start.py           환경 준비, 모델 파일 찾기, 앱 실행
├─ app.py             앱 화면 (PySide6)
├─ analysis.py        계산부 (웹 버전 server.py 의 계산 코드와 같음)
├─ chesswdl/          Drive chess/chesswdl 에서 추론에 필요한 파일만 옮긴 사본
├─ ckpt/             LATEST.json + round04_2025-04.pt (모델 파일은 Release 의 zip 에 포함)
├─ results/calib.json 엔진 보정 파일 (Drive 의 것과 같은 값)
└─ fonts/             Hahmlet, Gothic A1 (SIL Open Font License, 라이선스 파일 포함)
```

## 문제가 생기면

- **"Python 3.9+ was not found"**: https://www.python.org 에서 Python을 설치하고 "Add python.exe to PATH"를 체크한 뒤 다시 실행하세요.
- **창이 안 뜨고 PySide6 오류가 날 때**: `_deps` 폴더를 지우고 다시 실행하세요.
- **다른 환경을 쓰게 하고 싶을 때**: `.runtime.json`을 지우면 다음 실행 때 다시 찾습니다.
- **GPU가 있는데 CPU로 돈다고 나올 때**: 다시 쓰는 환경의 PyTorch가 CPU 버전인 경우입니다. `.runtime.json`을 지우고, 이 폴더에 `.venv`를 새로 만들게 하려면 예전 `.venv`를 찾지 못하는 곳으로 옮긴 뒤 실행하세요.
- 자세한 오류는 검은 창에 출력됩니다.

## 해석 주의점

- 승률이 달라진 몫은 대부분 두 사람의 레이팅 차이에서 옵니다. 레이팅에 따라 수 선택이 달라지는지는 후보 표의 고를 확률 차이로 보세요.
- PGN의 플레이어 이름이 학습 vocab에 있으면 그 사람의 임베딩이 들어갑니다 (`predict.py`와 같은 방식).
- 직접 둔 수순은 시계 정보 없이 계산합니다. FEN 태그로 시작하는 대국과 변형 체스는 분석하지 않습니다.

## 모델에 대해

- 구조: 포지션별 Transformer 인코더(8층, d=256) + 대국 흐름 GRU(2층) + 승무패 head와 다음 수 head, 약 8.74M 파라미터
- 학습 데이터: Lichess Open Database (CC0) 의 blitz~rapid 대국, 2024-01과 2025-01~04 월별 덤프에서 달마다 20만 판씩 이어서 학습
- 검증셋 다음 수 Top-1 약 34%, 승무패 logloss 약 0.74
- 체크포인트에는 학습 데이터에 자주 나온 Lichess 사용자 48명의 이름과 임베딩이 들어 있습니다 (공개 데이터의 공개 아이디)
