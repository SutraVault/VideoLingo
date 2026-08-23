import copy
import os,sys
import glob
import re
import subprocess
from core.utils import *


YOUTUBE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.youtube.com/",
}

def sanitize_filename(filename):
    # Remove or replace illegal characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Ensure filename doesn't start or end with a dot or space
    filename = filename.strip('. ')
    # Use default name if filename is empty
    return filename if filename else 'video'

def update_ytdlp():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"])
        if 'yt_dlp' in sys.modules:
            del sys.modules['yt_dlp']
        rprint("[green]yt-dlp updated[/green]")
    except subprocess.CalledProcessError as e:
        rprint("[yellow]Warning: Failed to update yt-dlp: {e}[/yellow]")
    from yt_dlp import YoutubeDL
    return YoutubeDL

def _format_candidates(resolution):
    if resolution == 'best':
        return [
            'bestvideo+bestaudio/best',
            'best[ext=mp4]/best',
        ]
    return [
        f'bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]',
        f'best[ext=mp4][height<={resolution}]/best[height<={resolution}]',
    ]

def _build_ydl_opts(save_path, format_selector):
    ydl_opts = {
        'format': format_selector,
        'outtmpl': f'{save_path}/%(title)s.%(ext)s',
        'noplaylist': True,
        'writethumbnail': True,
        'postprocessors': [{'key': 'FFmpegThumbnailsConvertor', 'format': 'jpg'}],
        'http_headers': YOUTUBE_HEADERS,
        'retries': 10,
        'fragment_retries': 10,
        'extractor_retries': 3,
        'socket_timeout': 30,
        'merge_output_format': 'mp4',
    }

    cookies_path = str(load_key("youtube.cookies_path") or "").strip()
    if cookies_path and os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = cookies_path

    return ydl_opts

def _is_http_403_error(error):
    message = str(error)
    return "HTTP Error 403" in message or "Forbidden" in message

def download_video_ytdlp(url, save_path='output', resolution='1080'):
    os.makedirs(save_path, exist_ok=True)

    # Get YoutubeDL class after updating
    YoutubeDL = update_ytdlp()
    last_error = None
    for format_selector in _format_candidates(resolution):
        ydl_opts = _build_ydl_opts(save_path, format_selector)
        try:
            with YoutubeDL(copy.deepcopy(ydl_opts)) as ydl:
                ydl.download([url])
            last_error = None
            break
        except Exception as e:
            last_error = e
            if not _is_http_403_error(e):
                raise
            rprint("[yellow]yt-dlp got HTTP 403; retrying with a fallback format...[/yellow]")

    if last_error is not None:
        cookies_path = str(load_key("youtube.cookies_path") or "").strip()
        cookie_hint = (
            " Set a valid YouTube cookies.txt path in the sidebar's Youtube Settings."
            if not cookies_path or not os.path.exists(cookies_path)
            else " The configured YouTube cookies file was used, but YouTube still rejected the media URL."
        )
        raise RuntimeError(
            "YouTube rejected the video download with HTTP 403 Forbidden."
            + cookie_hint
            + " Try exporting fresh browser cookies for youtube.com, or upload the video file manually."
        ) from last_error
    
    # Check and rename files after download
    for file in os.listdir(save_path):
        if os.path.isfile(os.path.join(save_path, file)):
            filename, ext = os.path.splitext(file)
            new_filename = sanitize_filename(filename)
            if new_filename != filename:
                os.rename(os.path.join(save_path, file), os.path.join(save_path, new_filename + ext))

def find_video_files(save_path='output'):
    video_files = [file for file in glob.glob(save_path + "/*") if os.path.splitext(file)[1][1:].lower() in load_key("allowed_video_formats")]
    # change \\ to /, this happen on windows
    if sys.platform.startswith('win'):
        video_files = [file.replace("\\", "/") for file in video_files]
    video_files = [file for file in video_files if not file.startswith("output/output")]
    if len(video_files) != 1:
        raise ValueError(f"Number of videos found {len(video_files)} is not unique. Please check.")
    return video_files[0]

if __name__ == '__main__':
    # Example usage
    url = input('Please enter the URL of the video you want to download: ')
    resolution = input('Please enter the desired resolution (360/480/720/1080, default 1080): ')
    resolution = int(resolution) if resolution.isdigit() else 1080
    download_video_ytdlp(url, resolution=resolution)
    print(f"🎥 Video has been downloaded to {find_video_files()}")
