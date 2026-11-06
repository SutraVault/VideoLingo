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


def recover_with_context(decode, start, end, duration, threshold=2.5, attempts=3, realign=None, local_repair=True):
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
                recovered=repair_isolated_zero_words(
                    preserve_valid_decoder_timing(aligned,recovered,threshold))
                invalid=suspicious_windows(recovered,duration,threshold,padding=0)
                diagnostic.update(segments=recovered,invalid_windows=invalid,acoustic_realignment=True)
            except Exception as exc:
                diagnostic['realignment_error']=str(exc)
        diagnostics.append(diagnostic)
        if recovered and not invalid:
            return recovered,a,b,diagnostics
    # Broad conflict windows can span minutes. Re-decoding all of that audio
    # repeatedly fixes one word while destabilizing another. Retry only the
    # remaining defects of the final candidate, with bounded context/recursion.
    if local_repair and diagnostics and recovered and invalid:
        candidate=copy.deepcopy(recovered)
        local_diagnostics=[]
        repaired_start,repaired_end=a,b
        windows=suspicious_windows(candidate,duration,threshold,padding=3)
        for left,right in windows:
            # A local repair must be smaller than the already attempted crop.
            left=max(a,left);right=min(b,right)
            if right-left >= b-a:
                break
            fixed,lo,hi,details=recover_with_context(
                decode,left,right,duration,threshold,attempts,realign,local_repair=False)
            local_diagnostics.append({'start':left,'end':right,'attempts':details,'resolved':fixed is not None})
            if fixed is None:
                break
            candidate=replace_window(candidate,fixed,lo,hi)
            repaired_start=min(repaired_start,lo)
            repaired_end=max(repaired_end,hi)
        else:
            if not suspicious_windows(candidate,duration,threshold,padding=0):
                diagnostics[-1]['local_repairs']=local_diagnostics
                # Expanded local crops can extend beyond the broad candidate.
                return candidate,repaired_start,repaired_end,diagnostics
        diagnostics[-1]['local_repairs']=local_diagnostics
    return None,start,end,diagnostics


def preserve_valid_decoder_timing(aligned, decoded, threshold=2.5):
    """Restore valid decoder bounds only where acoustic alignment regressed.

    Match normalized character spans because aligners can join split tokens.
    Include touching neighbours when necessary; every restored decoder word
    must be valid and ordered, and the group cannot cross aligned neighbours.
    """
    def valid(word):
        return ('start' in word and 'end' in word
                and 0 < word['end']-word['start'] <= threshold*max(1,len(word.get('word','').split())))

    originals={}
    offset=0
    for segment in decoded:
        for word in segment.get('words',[]):
            end=offset+len(normalize(word.get('word','')))
            if end>offset and valid(word):
                originals[offset,end]=word
            offset=end
    result=copy.deepcopy(aligned)
    words=[w for s in result for w in s.get('words',[])]
    offset=0
    matches=[]
    for word in words:
        end=offset+len(normalize(word.get('word','')))
        matches.append(originals.get((offset,end)))
        offset=end
    for i,word in enumerate(words):
        original=matches[i]
        if valid(word) or original is None:
            continue
        left=right=i
        while left and matches[left] is not None and words[left-1].get('end',float('inf'))>matches[left]['start']:
            left-=1
        while right+1<len(words) and matches[right] is not None and words[right+1].get('start',float('-inf'))<matches[right]['end']:
            right+=1
        group=matches[left:right+1]
        if any(w is None for w in group):
            continue
        if any(a['end']>b['start'] for a,b in zip(group,group[1:])):
            continue
        if left and words[left-1].get('end',float('inf'))>group[0]['start']:
            continue
        if right+1<len(words) and words[right+1].get('start',float('-inf'))<group[-1]['end']:
            continue
        for target,source in zip(words[left:right+1],group):
            target['original_timing']={'start':target.get('start'),'end':target.get('end')}
            target.update(start=source['start'],end=source['end'],score=source.get('score',0),
                          timing_method='validated_decoder_fallback')
    for segment in result:
        timed=segment.get('words',[])
        if timed and all('start' in w and 'end' in w for w in timed):
            segment.update(start=timed[0]['start'],end=timed[-1]['end'])
    return result


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
