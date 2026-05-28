import platform
import subprocess

import cv2
import numpy as np
from rich.console import Console

from core._1_ytdlp import find_video_files
from core.asr_backend.audio_preprocess import normalize_audio_volume
from core.utils import *
from core.utils.models import *

console = Console()

DUB_VIDEO = "output/output_dub.mp4"
DUB_SUB_FILE = 'output/dub.srt'
DUB_AUDIO = 'output/dub.mp3'
SRC_SUB_FILE = 'output/src.srt'

SRC_FONT_SIZE = 15
TRANS_FONT_SIZE = 17
FONT_NAME = 'Arial'
TRANS_FONT_NAME = 'Microsoft YaHei'
if platform.system() == 'Linux':
    FONT_NAME = 'NotoSansCJK-Regular'
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
if platform.system() == 'Darwin':
    FONT_NAME = 'Arial Unicode MS'
    TRANS_FONT_NAME = 'Arial Unicode MS'

SRC_FONT_COLOR = '&HFFFFFF'
SRC_OUTLINE_COLOR = '&H000000'
SRC_OUTLINE_WIDTH = 1
SRC_SHADOW_COLOR = '&H80000000'
TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1 
TRANS_BACK_COLOR = '&H33000000'

WATERMARK_MARGIN = 24

def _escape_drawtext_text(text: str) -> str:
    """Escape user text for FFmpeg drawtext's text option."""
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(",", "\\,")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )

def _watermark_position_expr(position: str) -> tuple[str, str]:
    margin = WATERMARK_MARGIN
    positions = {
        "top_left": (str(margin), str(margin)),
        "top_right": (f"w-tw-{margin}", str(margin)),
        "bottom_left": (str(margin), f"h-th-{margin}"),
        "bottom_right": (f"w-tw-{margin}", f"h-th-{margin}"),
        "center": ("(w-tw)/2", "(h-th)/2"),
    }
    return positions.get(position, positions["top_right"])

def _build_text_watermark_filter() -> str:
    try:
        if not load_key("watermark.enabled"):
            return ""
        text = str(load_key("watermark.text")).strip()
        if not text:
            return ""
        opacity = max(0.0, min(1.0, float(load_key("watermark.opacity"))))
        font_size = max(12, min(96, int(load_key("watermark.font_size"))))
        x_expr, y_expr = _watermark_position_expr(load_key("watermark.position"))
        escaped_text = _escape_drawtext_text(text)
        return (
            f",drawtext=font='{TRANS_FONT_NAME}':text='{escaped_text}':"
            f"x={x_expr}:y={y_expr}:fontsize={font_size}:"
            f"fontcolor=white@{opacity}:borderw=2:bordercolor=black@{opacity}"
        )
    except Exception as exc:
        rprint(f"[yellow]Watermark disabled because config is invalid: {exc}[/yellow]")
        return ""

def merge_video_audio():
    """Merge video and audio, and reduce video volume"""
    VIDEO_FILE = find_video_files()
    background_file = _BACKGROUND_AUDIO_FILE
    watermark_filter = _build_text_watermark_filter()
    
    if not load_key("burn_subtitles") and not watermark_filter:
        rprint("[bold yellow]Warning: A 0-second black video will be generated as a placeholder as subtitles are not burned in.[/bold yellow]")

        # Create a black frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(DUB_VIDEO, fourcc, 1, (1920, 1080))
        out.write(frame)
        out.release()

        rprint("[bold green]Placeholder video has been generated.[/bold green]")
        return

    # Normalize dub audio
    normalized_dub_audio = 'output/normalized_dub.wav'
    normalize_audio_volume(DUB_AUDIO, normalized_dub_audio)
    
    # Merge video and audio with translated subtitles
    video = cv2.VideoCapture(VIDEO_FILE)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")
    
    video_filter = (
        f'[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,'
        f'pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2'
    )
    if load_key("burn_subtitles"):
        video_filter += (
            f",subtitles={SRC_SUB_FILE}:force_style='FontSize={SRC_FONT_SIZE},"
            f"FontName={FONT_NAME},PrimaryColour={SRC_FONT_COLOR},"
            f"OutlineColour={SRC_OUTLINE_COLOR},OutlineWidth={SRC_OUTLINE_WIDTH},"
            f"ShadowColour={SRC_SHADOW_COLOR},BorderStyle=1',"
            f"subtitles={DUB_SUB_FILE}:force_style='FontSize={TRANS_FONT_SIZE},"
            f"FontName={TRANS_FONT_NAME},PrimaryColour={TRANS_FONT_COLOR},"
            f"OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
            f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV=27,BorderStyle=4'"
        )
    video_filter += f"{watermark_filter}[v]"
    
    cmd = [
        'ffmpeg', '-y', '-i', VIDEO_FILE, '-i', background_file, '-i', normalized_dub_audio,
        '-filter_complex',
        f'{video_filter};'
        f'[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=3[a]'
    ]

    if load_key("ffmpeg_gpu"):
        rprint("[bold green]Using GPU acceleration...[/bold green]")
        cmd.extend(['-map', '[v]', '-map', '[a]', '-c:v', 'h264_nvenc'])
    else:
        cmd.extend(['-map', '[v]', '-map', '[a]'])
    
    cmd.extend(['-c:a', 'aac', '-b:a', '96k', DUB_VIDEO])
    
    subprocess.run(cmd)
    rprint(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")

if __name__ == '__main__':
    merge_video_audio()
