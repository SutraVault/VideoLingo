from pathlib import Path
import ast
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
import wave

import requests

from core.utils import *


SERVER_PROCESS = None
_SHARED_REFERENCE_CACHE = {}
_DURATION_STATS = {"first_passes": 0, "predicted_first_passes": 0, "corrective_retries": 0}
_DURATION_CALIBRATION = []


def reset_indextts_duration_stats():
    for key in _DURATION_STATS:
        _DURATION_STATS[key] = 0
    _DURATION_CALIBRATION.clear()


def get_indextts_duration_stats():
    stats = dict(_DURATION_STATS)
    stats["calibration_ratio"] = (
        round(statistics.median(_DURATION_CALIBRATION), 3)
        if _DURATION_CALIBRATION else 1.0
    )
    return stats


def _active_settings():
    """Return shared settings merged with the selected engine profile."""
    settings = dict(load_key("indextts"))
    version = str(settings.get("version", "2"))
    profile_key = "v2_5" if version == "2.5" else "v2"
    profile = settings.get(profile_key, {}) or {}
    settings.update(profile)
    settings["version"] = version
    settings["profile_key"] = profile_key
    return settings


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
    is_v25 = settings.get("version") == "2.5"
    if is_v25:
        server_script = Path(__file__).resolve().with_name("indextts25_server.py")
    else:
        server_script = repo_dir / "indextts_server.py"
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
        cmd = [str(uv_exe), "run", "python", str(server_script)]
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
        cmd = [str(python_exe), str(server_script)]

    model_dir = _resolve_from_project(settings.get("model_dir", str(repo_dir / "checkpoints")))
    cmd.extend([
        "--host",
        str(settings.get("host", "127.0.0.1")),
        "--port",
        str(settings.get("port", 9871)),
        "--model_dir",
        str(model_dir),
    ])

    if is_v25 and settings.get("bf16", False):
        cmd.append("--bf16")
    elif settings.get("fp16", False):
        cmd.append("--fp16")
    if settings.get("cuda_kernel", False):
        cmd.append("--cuda_kernel")
    if settings.get("deepspeed", False):
        cmd.append("--deepspeed")
    if settings.get("accel", False):
        cmd.append("--accel")
    if settings.get("torch_compile", False):
        cmd.append("--torch_compile")
    if is_v25 and (settings.get("use_qwen_emo", False) or settings.get("use_emo_text", False)):
        cmd.append("--use_qwen_emo")

    return repo_dir, cmd


def start_indextts_server():
    global SERVER_PROCESS
    settings = _active_settings()
    host = str(settings.get("host", "127.0.0.1"))
    port = int(settings.get("port", 9871))
    api_url = f"http://{host}:{port}/tts"
    ping_url = api_url.rsplit("/", 1)[0] + "/ping"

    if _is_port_open(host, port):
        try:
            response = requests.get(ping_url, timeout=3)
            if response.status_code == 200:
                version_matches = settings["version"] != "2.5" or response.json().get("version") == "2.5"
                if version_matches:
                    return
        except (requests.RequestException, ValueError):
            pass

    repo_dir, cmd = _build_server_command(settings)
    if not repo_dir.exists():
        raise FileNotFoundError(f"IndexTTS {settings['version']} repository not found: {repo_dir}")
    model_dir = _resolve_from_project(settings.get("model_dir", repo_dir / "checkpoints"))
    if not (model_dir / "config.yaml").exists():
        raise FileNotFoundError(f"IndexTTS {settings['version']} model config not found: {model_dir / 'config.yaml'}")
    if settings["version"] == "2" and not (repo_dir / "indextts_server.py").exists():
        raise FileNotFoundError(f"IndexTTS server file not found: {repo_dir / 'indextts_server.py'}")

    rprint("[bold yellow]Initializing IndexTTS server...[/bold yellow]")
    log_dir = Path.cwd() / "output" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"indextts_{settings['version'].replace('.', '_')}_server.log"
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
    cache_key = (str(refers_dir.resolve()), min_duration, max_duration)
    cached_path = _SHARED_REFERENCE_CACHE.get(cache_key)
    if cached_path is not None and cached_path.exists():
        return cached_path
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
            _SHARED_REFERENCE_CACHE[cache_key] = path
            return path

    if candidates:
        path, duration = max(candidates, key=lambda item: item[1])
        if path != fallback:
            rprint(
                f"[yellow]No IndexTTS reference met min_refer_duration={min_duration:.1f}s; "
                f"using longest reference {path.name} ({duration:.2f}s).[/yellow]"
            )
        _SHARED_REFERENCE_CACHE[cache_key] = path
        return path

    _SHARED_REFERENCE_CACHE[cache_key] = fallback
    return fallback


def _reference_audio_for(number):
    settings = _active_settings()
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


def indextts_tts(text, save_path, ref_audio_path, duration_factor=None):
    settings = _active_settings()
    api_url = f"http://{settings.get('host', '127.0.0.1')}:{settings.get('port', 9871)}/tts"
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
        "interval_silence": settings.get("interval_silence", 80),
        "max_text_tokens_per_segment": settings.get("max_text_tokens_per_segment", 120),
    }
    if settings["version"] == "2.5":
        payload.update({
            "lang": settings.get("language", "ZH"),
            "duration_factor": (
                settings.get("duration_factor", 1.0)
                if duration_factor is None else duration_factor
            ),
            "text_normalization": settings.get("text_normalization", True),
        })
    response = requests.post(api_url, json=payload, timeout=int(settings.get("request_timeout", 600)))
    if response.status_code != 200:
        raise RuntimeError(f"IndexTTS request failed: HTTP {response.status_code}, {response.text}")

    result = response.json()
    if not result.get("ok"):
        raise RuntimeError(f"IndexTTS request failed: {result.get('error', 'unknown error')}")
    return True


def _line_duration_context(number, save_as, task_df):
    """Return this line's allocated timeline and estimated natural duration."""
    if task_df is None or not hasattr(task_df, "loc"):
        return None, None
    rows = task_df.loc[task_df["number"] == number]
    if rows.empty:
        return None, None
    row = rows.iloc[0]
    available = float(row.get("tol_dur", row.get("duration", 0)) or 0)
    if available <= 0:
        return None, None

    lines = row.get("lines", [])
    if isinstance(lines, str):
        try:
            lines = ast.literal_eval(lines)
        except (SyntaxError, ValueError):
            lines = [lines]
    if not isinstance(lines, (list, tuple)) or not lines:
        return available, float(row.get("est_dur", 0) or 0) or None

    match = re.search(r"_(\d+)_temp$", Path(save_as).stem)
    line_index = int(match.group(1)) if match else 0
    if line_index >= len(lines):
        return None, None

    # Punctuation contributes a small pause but should not dominate allocation.
    weights = [max(1, len(re.sub(r"[\s，。！？、；：,.!?;:]", "", str(line)))) for line in lines]
    weight_ratio = weights[line_index] / sum(weights)
    estimated_total = float(row.get("est_dur", 0) or 0)
    estimated_line = estimated_total * weight_ratio if estimated_total > 0 else None
    return available * weight_ratio, estimated_line


def _predicted_duration_factor(settings, estimated_duration, target_duration):
    """Choose a useful first-pass factor from the task duration estimate."""
    base = float(settings.get("duration_factor", 1.0))
    auto = settings.get("auto_duration", {}) or {}
    if not auto.get("enabled", False) or not estimated_duration or not target_duration:
        return base
    native_fit_speed = float(auto.get("native_fit_speed", 1.0))
    target = target_duration * float(auto.get("target_fill_ratio", 0.95)) * native_fit_speed
    threshold = float(auto.get("overflow_threshold", 0.08))
    calibration = statistics.median(_DURATION_CALIBRATION) if _DURATION_CALIBRATION else 1.0
    calibrated_estimate = estimated_duration * calibration
    if calibrated_estimate <= target * (1 + threshold):
        return base
    factor = base * target / calibrated_estimate
    factor = max(float(auto.get("min_factor", 0.75)), factor)
    factor = min(float(auto.get("max_factor", base)), factor)
    return round(factor, 3)


def _record_duration_calibration(estimated_duration, factor, actual_duration):
    """Learn how IndexTTS duration compares with the generic text estimator."""
    if not estimated_duration or estimated_duration <= 0 or factor <= 0 or actual_duration <= 0:
        return
    ratio = actual_duration / factor / estimated_duration
    # Reject pathological samples while retaining realistic model variation.
    if 0.5 <= ratio <= 3.5:
        _DURATION_CALIBRATION.append(ratio)
        del _DURATION_CALIBRATION[:-20]


def _auto_duration_factor(settings, first_duration, target_duration, current_factor=None):
    auto = settings.get("auto_duration", {}) or {}
    if not auto.get("enabled", False) or first_duration <= 0 or not target_duration:
        return None
    fill_ratio = float(auto.get("target_fill_ratio", 0.95))
    target = target_duration * fill_ratio * float(auto.get("native_fit_speed", 1.0))
    overflow_threshold = float(auto.get("overflow_threshold", 0.08))
    if first_duration <= target * (1 + overflow_threshold):
        return None

    base = float(settings.get("duration_factor", 1.0)) if current_factor is None else float(current_factor)
    factor = base * target / first_duration
    factor = max(float(auto.get("min_factor", 0.75)), factor)
    factor = min(float(auto.get("max_factor", base)), factor)
    if abs(factor - base) < 0.02:
        return None
    return round(factor, 3)


def indextts_tts_for_videolingo(text, save_as, number, task_df):
    start_indextts_server()
    settings = _active_settings()
    ref_audio_path = _reference_audio_for(number)
    try:
        target_duration, estimated_duration = _line_duration_context(number, save_as, task_df)
        first_factor = _predicted_duration_factor(settings, estimated_duration, target_duration)
        _DURATION_STATS["first_passes"] += 1
        if abs(first_factor - float(settings.get("duration_factor", 1.0))) >= 0.02:
            _DURATION_STATS["predicted_first_passes"] += 1
        result = indextts_tts(text, save_as, ref_audio_path, duration_factor=first_factor)
        if settings["version"] == "2.5":
            first_duration = _wav_duration(Path(save_as))
            _record_duration_calibration(estimated_duration, first_factor, first_duration)
            factor = _auto_duration_factor(settings, first_duration, target_duration, first_factor)
            if factor is not None:
                _DURATION_STATS["corrective_retries"] += 1
                rprint(
                    f"[cyan]IndexTTS2.5 subtitle {number}: {first_duration:.2f}s exceeds "
                    f"{target_duration:.2f}s allocation; regenerating with duration_factor={factor:.3f}.[/cyan]"
                )
                result = indextts_tts(text, save_as, ref_audio_path, duration_factor=factor)
        return result
    except Exception:
        if int(_active_settings().get("refer_mode", 2)) == 3:
            fallback = Path.cwd() / "output/audio/refers/1.wav"
            _ensure_reference_audio(fallback)
            rprint("[yellow]IndexTTS failed with per-line reference audio; retrying with the first reference audio.[/yellow]")
            return indextts_tts(text, save_as, fallback)
        raise


def regenerate_indextts_to_duration(text, save_as, number, max_duration):
    """Regenerate one cached IndexTTS2.5 line to a bounded native duration."""
    start_indextts_server()
    settings = _active_settings()
    if settings["version"] != "2.5":
        raise RuntimeError("Targeted duration regeneration requires IndexTTS 2.5")

    ref_audio_path = _reference_audio_for(number)
    auto = settings.get("auto_duration", {}) or {}
    factor = float(auto.get("min_factor", 0.75))
    emergency_min = float(auto.get("emergency_min_factor", 0.5))
    max_duration = float(max_duration)

    for attempt in range(2):
        indextts_tts(text, save_as, ref_audio_path, duration_factor=factor)
        duration = _wav_duration(Path(save_as))
        if duration <= max_duration:
            return duration, factor
        next_factor = factor * max_duration / duration * 0.98
        next_factor = max(emergency_min, min(factor - 0.02, next_factor))
        if next_factor >= factor or (factor <= emergency_min and attempt > 0):
            break
        factor = round(next_factor, 3)

    return _wav_duration(Path(save_as)), factor
