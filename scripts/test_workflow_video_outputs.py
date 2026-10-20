"""Render tiny synthetic videos through the real FFmpeg output functions."""
import os,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from core import _7_sub_into_vid as original
from core import _12_dub_to_vid as dubbed
from core.st_utils import handoff_alarm

ROOT=Path(__file__).resolve().parents[1]

class VideoOutputTests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'work/workflow-video-tests';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name);self.previous=Path.cwd();os.chdir(self.path)
        self.addCleanup(lambda:os.chdir(self.previous))
        (self.path/'output/audio').mkdir(parents=True)
        def ffmpeg(args):subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y',*args],check=True,capture_output=True)
        ffmpeg(['-f','lavfi','-i','color=c=blue:s=320x180:r=12','-f','lavfi','-i','sine=frequency=440:sample_rate=16000','-t','1','-c:v','libx264','-c:a','aac','source.mp4'])
        for name in ('output/audio/background.mp3','output/dub.mp3'):
            ffmpeg(['-f','lavfi','-i','sine=frequency=200:sample_rate=16000','-t','1',name])
        for name in ('src.srt','trans.srt','dub.srt'):
            (self.path/'output'/name).write_text('1\n00:00:00,000 --> 00:00:00,900\nTest caption\n',encoding='utf-8')

    def duration(self,path):
        return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(path)]))

    def test_original_modes_preserve_real_video_and_original_audio(self):
        for mode in ('translation','bilingual'):
            with patch.object(original,'find_video_files',return_value='source.mp4'),patch.object(original,'load_key',return_value=False):
                original.merge_subtitles_to_video(force_burn=True,subtitle_mode=mode)
            self.assertGreater(self.duration('output/output_sub.mp4'),.8)
            streams=subprocess.check_output(['ffprobe','-v','error','-select_streams','a','-show_entries','stream=codec_type','-of','csv=p=0','output/output_sub.mp4'])
            self.assertIn(b'audio',streams)

    def test_all_dub_modes_including_none_generate_full_length_video(self):
        for mode in ('none','translation','bilingual'):
            with patch.object(dubbed,'find_video_files',return_value='source.mp4'),patch.object(dubbed,'load_key',return_value=False),patch.object(handoff_alarm,'play_completion_alarm'):
                dubbed.merge_video_audio(subtitle_mode=mode)
            self.assertGreater(self.duration('output/output_dub.mp4'),.8)

if __name__=='__main__':unittest.main()
