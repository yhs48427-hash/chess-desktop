"""체스 승률 예측 데스크톱 앱 시작 스크립트. run.bat 이 이 파일을 실행합니다.

처음 실행할 때만 하는 일
  1) 실행 환경: torch, python-chess, numpy 가 이미 깔린 Python 을 찾아서 다시 씁니다
     (예: 예전 chess_app 폴더의 .venv). 찾으면 PyTorch 를 다시 받지 않고, 화면용 PySide6 만
     이 폴더의 _deps 에 따로 설치합니다 (그 환경 자체는 건드리지 않음).
     못 찾으면 이 폴더에 .venv 를 만들고 모두 설치합니다 (NVIDIA GPU가 있으면 CUDA 버전, 약 2.5GB).
  2) 학습된 모델 파일: ckpt/ 에 없으면 PC 에서 찾아 복사합니다
     (예전 chess_app\\ckpt, 다운로드 폴더, Google Drive 데스크톱 폴더, 바탕 화면\\CHESS 아래).
     그래도 없으면 Drive 파일 페이지를 열고, 다운로드 폴더에 받아지는 대로 가져옵니다.
그다음부터는 바로 앱 창을 띄웁니다. 추가 옵션은 app.py 로 넘어갑니다. 예) run.bat --device cpu
"""
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import webbrowser
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
IS_WIN = os.name == "nt"
VENV = os.path.join(ROOT, ".venv")
VENV_PY = os.path.join(VENV, "Scripts", "python.exe") if IS_WIN else os.path.join(VENV, "bin", "python")
DEPS = os.path.join(ROOT, "_deps")                 # 다른 환경을 빌려 쓸 때 PySide6 를 넣는 곳
RUNTIME = os.path.join(ROOT, ".runtime.json")      # 빌려 쓰는 Python 경로 기록

DRIVE_FILE_IDS = {"round04_2025-04.pt": "1k4TUYix1oXuMua1usLe1DGVFJiLFzT_8"}
DRIVE_CKPT_FOLDER = "https://drive.google.com/drive/folders/1aZJk-L2rYiIse-Dx63Uu-MbgjN0bByyw"

CORE = ("torch", "chess", "numpy")
GUI = "PySide6"
GUI_PIP = "PySide6-Essentials"
TORCH_CUDA_TAGS = ("cu126", "cu128", "cu124")
MIN_CKPT_BYTES = 1_000_000
WAIT_DOWNLOAD_SEC = 20 * 60
SKIP_DIRS = {"node_modules", ".git", "site-packages", "Lib", "shards", "__pycache__", "AppData", "_deps"}

if os.path.isdir(DEPS):
    sys.path.insert(0, DEPS)


def say(msg=""):
    print(msg, flush=True)


def fail(msg):
    say()
    say(msg)
    sys.exit(1)


def same_path(a, b):
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def same_exe(a, b):
    # 실행 파일은 실제 경로로 풀지 않고 비교 (리눅스 venv 의 python 은 원래 python 의 링크라서)
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def child_env(**extra):
    env = dict(os.environ, CHESSAPP_CHILD="1", **extra)
    if os.path.isdir(DEPS):
        env["PYTHONPATH"] = DEPS + os.pathsep + env.get("PYTHONPATH", "")
    return env


def rerun_with(py, **extra):
    sys.exit(subprocess.call([py, os.path.abspath(__file__), *sys.argv[1:]], env=child_env(**extra)))


# ---------------------------------------------------------------- 1) 실행 환경
def missing(mods):
    """설치 여부만 확인 (불러오지 않음: numpy 없이 torch 를 먼저 불러오면 그 프로세스에서 torch-numpy 연결이 끊김)."""
    return [m for m in mods if importlib.util.find_spec(m) is None]


def pip(py, *args):
    return subprocess.call([py, "-m", "pip", "install", "--disable-pip-version-check", *args])


def install_core(py, miss):
    if "torch" in miss:
        gpu = shutil.which("nvidia-smi") is not None
        ok = False
        for tag in (TORCH_CUDA_TAGS if gpu else ("cpu",)):
            say(f"PyTorch 설치 중 ({'CUDA ' + tag[2:] if tag != 'cpu' else 'CPU'} 버전)...")
            if pip(py, "torch", "--index-url", f"https://download.pytorch.org/whl/{tag}") == 0:
                ok = True
                break
        if not ok:
            say("PyTorch 공식 저장소에서 받지 못해 PyPI에서 받습니다.")
            ok = pip(py, "torch") == 0
        if not ok:
            fail("PyTorch 설치에 실패했습니다. 위의 pip 오류를 확인하세요. 인터넷 연결과 디스크 공간(약 5GB)이 필요합니다.")
    rest = [m for m in miss if m in ("chess", "numpy")]
    if rest and pip(py, *rest) != 0:
        fail("python-chess / numpy 설치에 실패했습니다. 위의 pip 오류를 확인하세요.")


def install_gui(py, into_deps):
    say("화면 라이브러리(PySide6) 설치 중...")
    args = ["--target", DEPS, "--upgrade", GUI_PIP] if into_deps else [GUI_PIP]
    if pip(py, *args) != 0:
        fail("PySide6 설치에 실패했습니다. 위의 pip 오류를 확인하세요.")


def venv_dirs():
    """예전에 만든 .venv 후보: 이 폴더의 위쪽 폴더들과 바탕 화면\\CHESS 아래 (깊이 3까지)."""
    home = os.path.expanduser("~")
    roots = [os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT)),
             os.path.join(home, "Desktop", "CHESS"), os.path.join(home, "OneDrive", "Desktop", "CHESS"),
             os.path.join(home, "OneDrive", "바탕 화면", "CHESS")]
    seen, out = set(), []
    for root in roots:
        if not os.path.isdir(root):
            continue
        base_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, _ in os.walk(root):
            if dirpath.rstrip(os.sep).count(os.sep) - base_depth >= 3:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            if ".venv" in dirnames:
                v = os.path.join(dirpath, ".venv")
                key = os.path.normcase(os.path.realpath(v))
                if key not in seen and not same_path(v, VENV):
                    seen.add(key)
                    out.append(v)
                dirnames.remove(".venv")
    return out


def find_existing_python():
    best = None
    for v in venv_dirs():
        py = os.path.join(v, "Scripts", "python.exe") if IS_WIN else os.path.join(v, "bin", "python")
        if not os.path.exists(py):
            continue
        say(f"  확인 중: {py}")
        try:
            r = subprocess.run([py, "-c", "import numpy, chess, torch; print(torch.cuda.is_available())"],
                               capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0:
            cuda = r.stdout.strip().endswith("True")
            if cuda:
                return py
            best = best or py
    return best


def ensure_runtime():
    is_child = bool(os.environ.get("CHESSAPP_CHILD"))
    in_our_venv = same_path(sys.prefix, VENV)
    # 예전에 고른 Python 이 있으면 그걸로 실행
    if not is_child and not in_our_venv and os.path.exists(RUNTIME):
        try:
            py = json.load(open(RUNTIME, encoding="utf-8"))["python"]
        except (OSError, ValueError, KeyError):
            py = None
        if py and os.path.exists(py) and not same_exe(py, sys.executable):
            rerun_with(py)
    core_miss, gui_miss = missing(CORE), missing([GUI])
    if not core_miss and not gui_miss:
        return
    if os.environ.get("CHESSAPP_RESTARTED"):
        fail(f"설치했지만 찾지 못한 패키지가 있습니다: {', '.join(core_miss + gui_miss)}. "
             f"{VENV} 와 {DEPS} 폴더를 지우고 다시 실행해 보세요.")
    if not core_miss:
        # torch 가 있는 환경 (빌려 쓰는 환경이면 _deps 에, 우리 .venv 면 그 안에 설치)
        install_gui(sys.executable, into_deps=not in_our_venv)
        say("설치를 마쳤습니다. 다시 시작합니다.")
        rerun_with(sys.executable, CHESSAPP_RESTARTED="1")
    if in_our_venv:
        install_core(sys.executable, core_miss)
        if gui_miss:
            install_gui(sys.executable, into_deps=False)
        say("설치를 마쳤습니다. 다시 시작합니다.")
        rerun_with(sys.executable, CHESSAPP_RESTARTED="1")
    say("PyTorch가 깔린 기존 환경을 찾는 중입니다 (예전 chess_app 의 .venv 등).")
    found = find_existing_python()
    if found:
        say(f"기존 환경을 다시 씁니다: {found}")
        json.dump({"python": found}, open(RUNTIME, "w", encoding="utf-8"))
        rerun_with(found)
    say("쓸 만한 기존 환경이 없어 이 폴더에 전용 환경(.venv)을 만듭니다. 처음 한 번만 하고, PyTorch를 받느라 몇 분 걸립니다.")
    if not os.path.exists(VENV_PY):
        rc = subprocess.call([sys.executable, "-m", "venv", VENV])
        if rc != 0 or not os.path.exists(VENV_PY):
            fail("가상환경(.venv)을 만들지 못했습니다. Python 설치를 확인하세요.")
    rerun_with(VENV_PY)


# ---------------------------------------------------------------- 2) 모델 파일
def is_valid_ckpt(path):
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= MIN_CKPT_BYTES and zipfile.is_zipfile(path)
    except OSError:
        return False


def downloads_dir():
    if IS_WIN:
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                 r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders")
            val, _ = winreg.QueryValueEx(key, "{374DE290-123F-4565-9164-39C4925E467B}")
            path = os.path.expandvars(val)
            if os.path.isdir(path):
                return path
        except OSError:
            pass
    return os.path.join(os.path.expanduser("~"), "Downloads")


def local_drive_roots():
    if not IS_WIN:
        return []
    import ctypes
    import string
    k32 = ctypes.windll.kernel32
    mask = k32.GetLogicalDrives()
    return [f"{c}:\\" for i, c in enumerate(string.ascii_uppercase)
            if mask & (1 << i) and k32.GetDriveTypeW(f"{c}:\\") in (3, 4)]


def find_ckpt(name):
    home = os.path.expanduser("~")
    paths = [os.path.join(r, d, "chess", "ckpt", name) for r in local_drive_roots() for d in ("내 드라이브", "My Drive")]
    paths += [os.path.join(home, d, "chess", "ckpt", name) for d in ("Google Drive", "내 드라이브", "My Drive")]
    stem = os.path.splitext(name)[0]
    paths += sorted(glob.glob(os.path.join(downloads_dir(), glob.escape(stem) + "*.pt")),
                    key=lambda p: os.path.getmtime(p), reverse=True)
    for p in paths:
        if is_valid_ckpt(p):
            return p
    # 예전 chess_app 등: 이 폴더의 위쪽 폴더와 바탕 화면\CHESS 아래에서 ckpt\<name> 찾기 (깊이 4까지)
    roots = [os.path.dirname(ROOT), os.path.dirname(os.path.dirname(ROOT)), os.path.join(home, "Desktop", "CHESS"),
             os.path.join(home, "OneDrive", "Desktop", "CHESS"), os.path.join(home, "OneDrive", "바탕 화면", "CHESS")]
    for root in roots:
        if not os.path.isdir(root):
            continue
        base = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath.rstrip(os.sep).count(os.sep) - base >= 4:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and d != ".venv"]
            if name in filenames and not same_path(dirpath, os.path.join(ROOT, "ckpt")):
                p = os.path.join(dirpath, name)
                if is_valid_ckpt(p):
                    return p
    return None


def place(src, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = target + ".copying"
    shutil.copyfile(src, tmp)
    os.replace(tmp, target)
    say(f"모델 파일을 가져왔습니다: {src}")


def ensure_checkpoint():
    with open(os.path.join(ROOT, "ckpt", "LATEST.json"), encoding="utf-8") as f:
        rel = json.load(f)["ckpt"]
    target = os.path.join(ROOT, *rel.split("/"))
    name = os.path.basename(target)
    if is_valid_ckpt(target):
        return
    src = find_ckpt(name)
    if src:
        return place(src, target)
    fid = DRIVE_FILE_IDS.get(name)
    url = f"https://drive.google.com/file/d/{fid}/view" if fid else DRIVE_CKPT_FOLDER
    dl = downloads_dir()
    say(f"학습된 모델 파일({name}, 약 35MB)을 PC에서 찾지 못했습니다.")
    say("브라우저에 Google Drive 파일 페이지를 엽니다. 오른쪽 위의 다운로드 버튼을 눌러 주세요.")
    say(f"다운로드 폴더({dl})에 받아지는 대로 자동으로 가져옵니다. 직접 넣어도 됩니다: {target}")
    webbrowser.open(url)
    t0, last = time.time(), 0.0
    while time.time() - t0 < WAIT_DOWNLOAD_SEC:
        if is_valid_ckpt(target):
            return
        src = find_ckpt(name)
        if src:
            return place(src, target)
        if time.time() - last > 30:
            say("  모델 파일을 기다리는 중...")
            last = time.time()
        time.sleep(2)
    fail(f"{WAIT_DOWNLOAD_SEC // 60}분 동안 모델 파일을 찾지 못했습니다. "
         f"Drive의 chess/ckpt/{name} 을 받아 {target} 에 두고 다시 실행하세요.")


def main():
    if sys.version_info < (3, 9):
        fail(f"Python 3.9 이상이 필요합니다 (지금 {sys.version.split()[0]}).")
    os.chdir(ROOT)
    ensure_runtime()
    ensure_checkpoint()
    say("앱을 엽니다. 이 창을 닫으면 앱도 꺼집니다.")
    sys.path.insert(0, ROOT)
    import app
    sys.exit(app.main(sys.argv[1:]))


if __name__ == "__main__":
    main()
