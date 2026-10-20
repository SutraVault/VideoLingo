"""Select continuous speech references without changing their internal cadence."""
import hashlib
import json
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_silence


def prepare_reference(paths, cache_dir):
    """Return a validated, edge-trimmed reference; never splice out internal pauses."""
    paths = [Path(path) for path in paths if Path(path).is_file()]
    signature = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    key = hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:20]
    cache_dir = Path(cache_dir)
    output = cache_dir / f"continuous_v1_{key}.wav"
    report = output.with_suffix('.json')
    if output.exists() and report.exists():
        return output, json.loads(report.read_text(encoding='utf-8'))

    rejected = []
    short_candidate = None
    for path in paths:
        with path.open('rb') as source:
            audio = AudioSegment.from_wav(source)
        # Quiet recordings still need a threshold below their speech level.
        threshold = min(-40.0, audio.dBFS - 16)
        pauses = detect_silence(audio, min_silence_len=120, silence_thresh=threshold, seek_step=5)
        start = pauses[0][1] if pauses and pauses[0][0] == 0 else 0
        end = pauses[-1][0] if pauses and pauses[-1][1] == len(audio) else len(audio)
        if end <= start:
            rejected.append(f'{path.name}: no usable speech')
            continue
        # Retain a little padding at both edges, keeping originals untouched.
        clean = audio[max(0, start - 60):min(len(audio), end + 60)]
        inner = [(max(s, start), min(e, end)) for s, e in pauses if e > start and s < end]
        longest = max((e - s for s, e in inner), default=0)
        ratio = sum(e - s for s, e in inner) / (end - start)
        voiced_ms = end - start - sum(e - s for s, e in inner)
        details = {'source': str(path.resolve()), 'duration_ms': len(clean),
                   'trimmed_ms': len(audio) - len(clean), 'internal_silence_ratio': ratio,
                   'longest_internal_pause_ms': longest, 'voiced_ms': voiced_ms}
        if voiced_ms < 2500 or len(clean) > 15000 or longest > 600 or ratio > 0.25:
            rejected.append(f'{path.name}: voiced={voiced_ms}ms, pause={longest}ms, silence={ratio:.0%}')
            continue
        # Prefer a complete 6–15 second passage; short references remain useful
        # when no longer passage passes the same continuity checks.
        if len(clean) >= 6000:
            selected = clean, details
            break
        if short_candidate is None:
            short_candidate = clean, details
    else:
        selected = short_candidate
    if selected is None:
        raise ValueError('No clean, continuous reference speech found. Supply a clear speech sample. '
                         + '; '.join(rejected[:8]))
    clean, details = selected
    details['rejected_candidates'] = rejected
    cache_dir.mkdir(parents=True, exist_ok=True)
    clean.export(output, format='wav').close()
    report.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
    return output, details
