from pathlib import Path
import os
import shutil
import socket
import subprocess
import sys
import time
import wave

import requests

from core.utils import *


SERVER_PROCESS = None


def _resolve_from_project(path_value):
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _is_port_open(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def _build_server_command(settings):
    repo_dir = _resolve_from_project(settings.get("repo_dir", "../index-tts"))
    if settings.get("use_uv", True):
        uv_path = settings.get("uv_path", "")
        uv_exe = _resolve_from_project(uv_path) if uv_path else None
        if not uv_exe or not uv_exe.exists():
            uv_found = shutil.which("uv")
            uv_exe = Path(uv_found) if uv_found else None
        if not uv_exe or not uv_exe.exists():
            common_uv = Path.home() / "miniconda3" / "Scripts" / "uv.exe"
            uv_exe = common_uv if common_uv.exists() else None
        if not uv_exe or not uv_exe.exists():
            raise FileNotFoundError("uv executable not found. Set indextts.uv_path in config.yaml or disable indextts.use_uv.")
        cmd = [str(uv_exe), "run", "python", "indextts_server.py"]
    else:
        python_path = settings.get("python", "")
        if python_path and any(sep in str(python_path) for sep in ("/", "\\")):
            python_exe = _resolve_from_project(python_path)
        elif python_path:
            python_exe = Path(str(python_path))
        else:
            python_exe = repo_dir / ".venv" / "Scripts" / "python.exe"
        if python_exe.is_absolute() and not python_exe.exists():
            python_exe = Path(sys.executable)
        cmd = [str(python_exe), "indextts_server.py"]

    model_dir = _resolve_from_project(settings.get("model_dir", str(repo_dir / "checkpoints")))
    cmd.extend([
        "--host",
        str(settings.get("host", "127.0.0.1")),
        "--port",
        str(settings.get("port", 9871)),
        "--model_dir",
        str(model_dir),
    ])

    if settings.get("fp16", False):
        cmd.append("--fp16")
    if settings.get("cuda_kernel", False):
        cmd.append("--cuda_kernel")
    if settings.get("deepspeed", False):
        cmd.append("--deepspeed")
    if settings.get("accel", False):
        cmd.append("--accel")
    if settings.get("torch_compile", False):
        cmd.append("--torch_compile")

    return repo_dir, cmd


def start_indextts_server():
    global SERVER_PROCESS
    settings = load_key("indextts")
    host = str(settings.get("host", "127.0.0.1"))
    port = int(settings.get("port", 9871))
    api_url = settings.get("api_url", f"http://{host}:{port}/tts")
    ping_url = api_url.rsplit("/", 1)[0] + "/ping"

    if _is_port_open(host, port):
        try:
            response = requests.get(ping_url, timeout=3)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass

    repo_dir, cmd = _build_server_command(settings)
    if not (repo_dir / "indextts_server.py").exists():
        raise FileNotFoundError(f"IndexTTS server file not found: {repo_dir / 'indextts_server.py'}")

    rprint("[bold yellow]Initializing IndexTTS server...[/bold yellow]")
    log_dir = Path.cwd() / "output" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "indextts_server.log"
    env = os.environ.copy()
    env.pop("SSL_CERT_DIR", None)
    env.pop("SSL_CERT_FILE", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"\n\n--- Starting IndexTTS server: {' '.join(cmd)} ---\n")
    log_file.flush()
    if sys.platform == "win32":
        creationflags = 0
        if settings.get("show_console", False):
            creationflags = subprocess.CREATE_NEW_CONSOLE
        elif hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = subprocess.CREATE_NO_WINDOW
        SERVER_PROCESS = subprocess.Popen(
            cmd,
            cwd=repo_dir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    else:
        SERVER_PROCESS = subprocess.Popen(cmd, cwd=repo_dir, env=env, stdout=log_file, stderr=subprocess.STDOUT)

    start_time = time.time()
    while time.time() - start_time < int(settings.get("startup_timeout", 180)):
        if SERVER_PROCESS.poll() is not None:
            raise RuntimeError(
                f"IndexTTS server exited during startup with code {SERVER_PROCESS.returncode}. "
                f"Check the log: {log_path}"
            )
        try:
            response = requests.get(ping_url, timeout=5)
            if response.status_code == 200:
                rprint("[bold green]IndexTTS server is ready.[/bold green]")
                return
        except requests.RequestException:
            time.sleep(5)

    if SERVER_PROCESS.poll() is None:
        SERVER_PROCESS.terminate()
        try:
            SERVER_PROCESS.wait(timeout=10)
        except subprocess.TimeoutExpired:
            SERVER_PROCESS.kill()
    SERVER_PROCESS = None
    raise TimeoutError(f"IndexTTS server failed to start. Check the log: {log_path}")


def _ensure_reference_audio(ref_audio_path):
    if ref_audio_path.exists():
        return
    try:
        from core._9_refer_audio import extract_refer_audio_main

        rprint(f"[yellow]Reference audio missing, extracting: {ref_audio_path}[/yellow]")
        extract_refer_audio_main()
    except Exception as exc:
        rprint(f"[bold red]Failed to extract reference audio: {exc}[/bold red]")
        raise


def _wav_duration(path):
    try:
        with wave.open(str(path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if not frame_rate:
                return 0.0
            return wav_file.getnframes() / float(frame_rate)
    except (wave.Error, OSError):
        return 0.0


def _sort_reference_path(path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


def _shared_reference_audio(settings, current_dir):
    fallback = current_dir / "output/audio/refers/1.wav"
    _ensure_reference_audio(fallback)

    refers_dir = fallback.parent
    min_duration = float(settings.get("min_refer_duration", 0) or 0)
    max_duration = float(settings.get("max_refer_duration", 0) or 0)
    candidates = []

    for path in sorted(refers_dir.glob("*.wav"), key=_sort_reference_path):
        duration = _wav_duration(path)
        if duration <= 0:
            continue
        candidates.append((path, duration))
        if duration >= min_duration and (max_duration <= 0 or duration <= max_duration):
            if path != fallback:
                rprint(
                    f"[yellow]IndexTTS shared reference {fallback.name} is short "
                    f"({_wav_duration(fallback):.2f}s); using {path.name} ({duration:.2f}s).[/yellow]"
                )
            return path

    if candidates:
        path, duration = max(candidates, key=lambda item: item[1])
        if path != fallback:
            rprint(
                f"[yellow]No IndexTTS reference met min_refer_duration={min_duration:.1f}s; "
                f"using longest reference {path.name} ({duration:.2f}s).[/yellow]"
            )
        return path

    return fallback


def _reference_audio_for(number):
    settings = load_key("indextts")
    refer_mode = int(settings.get("refer_mode", 3))
    current_dir = Path.cwd()

    if refer_mode == 2:
        return _shared_reference_audio(settings, current_dir)
    elif refer_mode == 3:
        ref_audio_path = current_dir / f"output/audio/refers/{number}.wav"
    else:
        raise ValueError("Invalid indextts.refer_mode. Choose 2 or 3.")

    _ensure_reference_audio(ref_audio_path)
    if not ref_audio_path.exists() and refer_mode == 3:
        fallback = current_dir / "output/audio/refers/1.wav"
        _ensure_reference_audio(fallback)
        return fallback
    return ref_audio_path


def indextts_tts(text, save_path, ref_audio_path):
    settings = load_key("indextts")
    api_url = settings.get("api_url", f"http://{settings.get('host', '127.0.0.1')}:{settings.get('port', 9871)}/tts")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "text": text,
        "spk_audio_prompt": str(Path(ref_audio_path).resolve()),
        "output_path": str(save_path.resolve()),
        "emo_alpha": settings.get("emo_alpha", 1.0),
        "use_emo_text": settings.get("use_emo_text", False),
        "emo_text": settings.get("emo_text") or None,
        "use_random": settings.get("use_random", False),
        "max_text_tokens_per_segment": settings.get("max_text_tokens_per_segment", 120),
    }
    response = requests.post(api_url, json=payload, timeout=int(settings.get("request_timeout", 600)))
    if response.status_code != 200:
        raise RuntimeError(f"IndexTTS request failed: HTTP {response.status_code}, {response.text}")

    result = response.json()
    if not result.get("ok"):
        raise RuntimeError(f"IndexTTS request failed: {result.get('error', 'unknown error')}")
    return True


def indextts_tts_for_videolingo(text, save_as, number, task_df):
    start_indextts_server()
    ref_audio_path = _reference_audio_for(number)
    try:
        return indextts_tts(text, save_as, ref_audio_path)
    except Exception:
        if int(load_key("indextts.refer_mode")) == 3:
            fallback = Path.cwd() / "output/audio/refers/1.wav"
            _ensure_reference_audio(fallback)
            rprint("[yellow]IndexTTS failed with per-line reference audio; retrying with the first reference audio.[/yellow]")
            return indextts_tts(text, save_as, fallback)
        raise
