"""Dependency-free ASR recovery and alignment checks."""
import re
import copy


def normalize(text):
    return re.sub(r'[^\w]+', '', text.lower())


def merge_recovery(primary, recovery):
    merged = list(primary)
    added = []
    for candidate in recovery:
        local = [s for s in merged if min(s['end'], candidate['end']) > max(s['start'], candidate['start'])]
        text = normalize(candidate.get('text', ''))
        if not text:
            continue
        # Match only speech at the same time; repetitions elsewhere are valid.
        if any(text == normalize(s.get('text', '')) for s in local):
            continue
        # Replace an overlapping block only when recovery contains its text in
        # sequence, plus new words. Never discard a different overlapping claim.
        old = normalize(' '.join(s.get('text', '') for s in sorted(local, key=lambda s:s['start'])))
        if local and old and old in text:
            merged = [s for s in merged if s not in local]
        elif local:
            # Conflicting broad blocks require a word-timed independent retry.
            # Keep both candidates for alignment, rather than silently dropping
            # the only occurrence of potentially missing speech.
            candidate = dict(candidate, recovery_conflict=True)
        merged.append(candidate)
        added.append(candidate)
    return sorted(merged, key=lambda s:(s['start'],s['end'])), added


def suspicious_windows(segments, duration, max_word_seconds=2.5, padding=3.0):
    windows=[]
    for s in segments:
        if 'words' in s and normalize(s.get('text','')) and not s['words']:
            windows.append((max(0,s['start']-padding),min(duration,s['end']+padding)))
        if s.get('recovery_conflict'):
            windows.append((max(0,s['start']-padding), min(duration,s['end']+padding)))
        for w in s.get('words', []):
            if 'start' not in w or 'end' not in w:
                windows.append((max(0,s['start']-padding),min(duration,s['end']+padding)))
                continue
            # Permit phrase-sized aligner items, but flag a single long word.
            count=max(1,len(str(w.get('word','')).split()))
            if w['end'] <= w['start'] or w['end']-w['start'] > max_word_seconds*count:
                windows.append((max(0,w['start']-padding),min(duration,w['end']+padding)))
    merged=[]
    for a,b in sorted(windows):
        if merged and a<=merged[-1][1]: merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else: merged.append((a,b))
    return merged


def replace_window(segments, recovered, start, end):
    """Splice by word midpoint so surrounding parts of a long segment survive."""
    kept=[]
    for s in segments:
        if s['end']<=start or s['start']>=end:
            kept.append(s); continue
        if not s.get('words'):
            raise ValueError('Cannot safely splice an untimed overlapping segment')
        for side in (lambda m:m<start, lambda m:m>=end):
            words=[w for w in s['words'] if 'start' in w and 'end' in w and side((w['start']+w['end'])/2)]
            if words:
                kept.append(dict(s,start=words[0]['start'],end=words[-1]['end'],words=words,text=' '.join(w['word'].strip() for w in words)))
    return sorted(kept+recovered,key=lambda s:s['start'])


def recover_with_context(decode, start, end, duration, threshold=2.5, attempts=3, realign=None):
    """Retry unstable crop boundaries with bounded additional audio context.

    decode accepts clip bounds and returns word-timed segment dictionaries.
    Failed candidates are retained in the diagnostic, never silently accepted.
    """
    diagnostics=[]
    for attempt in range(attempts):
        padding=attempt*3.0
        a=max(0,start-padding);b=min(duration,end+padding)
        recovered=repair_isolated_zero_words(decode(a,b))
        invalid=suspicious_windows(recovered,duration,threshold,padding=0)
        diagnostic={'start':a,'end':b,'segments':recovered,'invalid_windows':invalid}
        if recovered and invalid and realign is not None:
            diagnostic['before_realignment']=copy.deepcopy(recovered)
            try:
                aligned=realign(copy.deepcopy(recovered),a,b)
                # Alignment may move timestamps, but must retain every word.
                expected=normalize(' '.join(w['word'] for s in recovered for w in s.get('words',[])))
                actual=normalize(' '.join(w['word'] for s in aligned for w in s.get('words',[])))
                if not aligned or actual != expected:
                    raise ValueError('Acoustic alignment changed or dropped recovery words')
                recovered=repair_isolated_zero_words(aligned)
                invalid=suspicious_windows(recovered,duration,threshold,padding=0)
                diagnostic.update(segments=recovered,invalid_windows=invalid,acoustic_realignment=True)
            except Exception as exc:
                diagnostic['realignment_error']=str(exc)
        diagnostics.append(diagnostic)
        if recovered and not invalid:
            return recovered,a,b,diagnostics
    return None,start,end,diagnostics


def repair_isolated_zero_words(segments):
    """Distribute a collapsed short token within its touching neighbours.

    Only an isolated, confident token in a short, continuously timed group is
    eligible. Consecutive missing words (possible hallucinations) remain errors.
    Every changed word retains its original bounds for audit.
    """
    result=copy.deepcopy(segments)
    for segment in result:
        words=segment.get('words',[])
        for i in range(1,len(words)-1):
            left,word,right=words[i-1:i+2]
            if not all('start' in w and 'end' in w for w in (left,word,right)):
                continue
            token=normalize(word.get('word',''))
            if (word['end'] != word['start'] or not 1<=len(token)<=8
                or word.get('score',0)<.6 or left['end']<=left['start']
                or right['end']<=right['start']):
                continue
            point=word['start']
            span=right['end']-left['start']
            if (abs(left['end']-point)>.02 or abs(right['start']-point)>.02
                or not .06<=span<=.8):
                continue
            weights=[max(1,len(normalize(w['word']))) for w in (left,word,right)]
            # Each token receives at least 20 ms; distribute the remaining
            # observed span by text length, without creating additional time.
            cursor=left['start'];total=sum(weights);observed_end=right['end']
            for w,weight in zip((left,word,right),weights):
                w.setdefault('original_timing',{'start':w['start'],'end':w['end']})
                w['start']=cursor
                cursor += .02+(span-.06)*weight/total
                w['end']=cursor
                w['timing_method']='bounded_token_interpolation'
            right['end']=observed_end
    return result
