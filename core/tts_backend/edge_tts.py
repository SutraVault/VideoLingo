from pathlib import Path
import edge_tts
from core.utils import *
import subprocess
import tempfile

from pydub import AudioSegment

# Available voices can be listed using edge-tts --list-voices command
# Common English voices:
# en-US-JennyNeural - Female
# en-US-GuyNeural - Male  
# en-GB-SoniaNeural - Female British
# Common Chinese voices:
# zh-CN-XiaoxiaoNeural - Female
# zh-CN-YunxiNeural - Male
# zh-CN-XiaoyiNeural - Female
def edge_tts(text, save_path):
    # Load settings from config file
    edge_set = load_key("edge_tts")
    voice = edge_set.get("voice", "en-US-JennyNeural")
    
    # Create output directory if it doesn't exist
    speech_file_path = Path(save_path)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)

    # edge-tts writes MP3 media regardless of the destination suffix. VideoLingo
    # stores its temporary speech clips as WAV, so writing directly to save_path
    # produces an MP3 file disguised as WAV and later FFmpeg decoding fails.
    temp_mp3 = None
    temp_wav = speech_file_path.with_suffix(speech_file_path.suffix + ".part")
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp3", prefix="videolingo_edge_", delete=False,
            dir=speech_file_path.parent,
        ) as temp_file:
            temp_mp3 = Path(temp_file.name)

        cmd = [
            "edge-tts", "--voice", voice, "--text", text,
            "--write-media", str(temp_mp3),
        ]
        subprocess.run(cmd, check=True)
        AudioSegment.from_file(temp_mp3, format="mp3").export(temp_wav, format="wav")
        temp_wav.replace(speech_file_path)
    finally:
        if temp_mp3 is not None:
            temp_mp3.unlink(missing_ok=True)
        temp_wav.unlink(missing_ok=True)

    print(f"Audio saved to {speech_file_path}")

if __name__ == "__main__":
    edge_tts("Today is a good day!", "edge_tts.wav")
