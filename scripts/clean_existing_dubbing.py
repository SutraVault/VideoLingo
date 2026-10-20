"""Back up and conservatively clean cached dubbing, without TTS or translation."""
import ast
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from core import _10_gen_audio as audio
from core import _11_merge_audio as merge
from core import _12_dub_to_vid as video


def main():
    if '--fit-only' in sys.argv:
        tasks = pd.read_excel(ROOT / 'output/audio/tts_tasks.xlsx')
        finish(tasks)
        return
    backup = ROOT / 'output' / 'dubbing_backups' / datetime.now().strftime('%Y%m%d_%H%M%S')
    backup.mkdir(parents=True, exist_ok=False)
    for relative in ('audio/tmp', 'audio/segs', 'audio/tts_tasks.xlsx',
                     'dub.mp3', 'dub.srt', 'normalized_dub.wav', 'output_dub.mp4'):
        source = ROOT / 'output' / relative
        if not source.exists():
            raise FileNotFoundError(source)
        target = backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    print(f'Backup complete: {backup}', flush=True)
    tasks = pd.read_excel(ROOT / 'output/audio/tts_tasks.xlsx')
    changes = []
    for index, row in tasks.iterrows():
        lines = ast.literal_eval(row['lines']) if isinstance(row['lines'], str) else row['lines']
        total = removed = 0.0
        for line_index in range(len(lines)):
            path = ROOT / 'output/audio/tmp' / f"{row['number']}_{line_index}_temp.wav"
            before, after, _ = audio.clean_abnormal_recovery_silence(str(path), 600, 220)
            total += after
            removed += before - after
            if before - after > 0.01:
                changes.append({'file': path.name, 'before': before, 'after': after})
        tasks.at[index, 'real_dur'] = total
        tasks.at[index, 'silence_removed'] = float(row.get('silence_removed', 0) or 0) + removed
    (backup / 'cleanup_report.json').write_text(json.dumps(changes, indent=2), encoding='utf-8')
    print(f'Cleaned {len(changes)} segments; removed {sum(c["before"]-c["after"] for c in changes):.2f}s of long silence', flush=True)
    finish(tasks)
    print(f'Completed. Original files preserved in {backup}', flush=True)


def finish(tasks):
    # Only run timeline fitting: no synthesis, preflight rewriting, or translation.
    with patch.object(audio, 'regenerate_oversized_indextts2_rows', side_effect=lambda df, *args: df), \
         patch.object(audio, 'regenerate_oversized_indextts_rows', side_effect=lambda df, *args: df):
        tasks = audio.merge_chunks(tasks)
    audio.save_audio_tasks(tasks)
    merge.merge_full_audio()
    video.merge_video_audio()
    print('Completed cached audio cleanup and video merge.', flush=True)


if __name__ == '__main__':
    main()
