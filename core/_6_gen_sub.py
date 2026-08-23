import pandas as pd
import os
import re
from difflib import SequenceMatcher
from rich.panel import Panel
from rich.console import Console
import autocorrect_py as autocorrect
from core.utils import *
from core.utils.models import *
console = Console()

SUBTITLE_OUTPUT_CONFIGS = [ 
    ('src.srt', ['Source']),
    ('trans.srt', ['Translation']),
    ('src_trans.srt', ['Source', 'Translation']),
    ('trans_src.srt', ['Translation', 'Source'])
]

AUDIO_SUBTITLE_OUTPUT_CONFIGS = [
    ('src_subs_for_audio.srt', ['Source']),
    ('trans_subs_for_audio.srt', ['Translation'])
]

def convert_to_srt_format(start_time, end_time):
    """Convert time (in seconds) to the format: hours:minutes:seconds,milliseconds"""
    def seconds_to_hmsm(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        milliseconds = int(seconds * 1000) % 1000
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

    start_srt = seconds_to_hmsm(start_time)
    end_srt = seconds_to_hmsm(end_time)
    return f"{start_srt} --> {end_srt}"

def remove_punctuation(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def safe_text(value):
    if pd.isna(value):
        return ''
    return str(value)

def show_difference(str1, str2):
    """Show the difference positions between two strings"""
    min_len = min(len(str1), len(str2))
    diff_positions = []
    
    for i in range(min_len):
        if str1[i] != str2[i]:
            diff_positions.append(i)
    
    if len(str1) != len(str2):
        diff_positions.extend(range(min_len, max(len(str1), len(str2))))
    
    print("Difference positions:")
    print(f"Expected sentence: {str1}")
    print(f"Actual match: {str2}")
    print("Position markers: " + "".join("^" if i in diff_positions else " " for i in range(max(len(str1), len(str2)))))
    print(f"Difference indices: {diff_positions}")

def fuzzy_find_sentence(full_words_str, clean_sentence, current_pos):
    """Find a near match for ASR spelling differences while preserving order."""
    sentence_len = len(clean_sentence)
    if sentence_len == 0:
        return None

    best_match = None
    best_score = 0
    max_start = min(len(full_words_str) - 1, current_pos + max(500, sentence_len * 5))
    min_candidate_len = max(1, int(sentence_len * 0.75))
    max_candidate_len = max(min_candidate_len, int(sentence_len * 1.25))

    for start in range(current_pos, max_start + 1):
        for candidate_len in range(min_candidate_len, max_candidate_len + 1):
            end = start + candidate_len
            if end > len(full_words_str):
                break
            candidate = full_words_str[start:end]
            score = SequenceMatcher(None, clean_sentence, candidate).ratio()
            if score > best_score:
                best_score = score
                best_match = (start, end, candidate, score)

    return best_match if best_match and best_score >= 0.78 else None

def get_sentence_timestamps(df_words, df_sentences, fallback_timestamps=None):
    time_stamp_list = []
    
    # Build complete string and position mapping
    full_words_str = ''
    position_to_word_idx = {}
    
    for idx, word in enumerate(df_words['text']):
        clean_word = remove_punctuation(safe_text(word).lower())
        start_pos = len(full_words_str)
        full_words_str += clean_word
        for pos in range(start_pos, len(full_words_str)):
            position_to_word_idx[pos] = idx
    
    current_pos = 0
    pending_empty_rows = []
    for idx, sentence in df_sentences['Source'].items():
        clean_sentence = remove_punctuation(safe_text(sentence).lower()).replace(" ", "")
        if not clean_sentence:
            time_stamp_list.append(None)
            pending_empty_rows.append(len(time_stamp_list) - 1)
            continue

        sentence_len = len(clean_sentence)
        sentence_start_pos = current_pos
        
        match_found = False
        while current_pos <= len(full_words_str) - sentence_len:
            if full_words_str[current_pos:current_pos+sentence_len] == clean_sentence:
                start_word_idx = position_to_word_idx[current_pos]
                end_word_idx = position_to_word_idx[current_pos + sentence_len - 1]
                timestamp = (
                    float(df_words['start'][start_word_idx]),
                    float(df_words['end'][end_word_idx])
                )
                if pending_empty_rows:
                    timestamp = _fill_pending_empty_timestamps(time_stamp_list, pending_empty_rows, timestamp)
                    pending_empty_rows = []
                time_stamp_list.append(timestamp)
                
                current_pos += sentence_len
                match_found = True
                break
            current_pos += 1
            
        if not match_found:
            fuzzy_match = fuzzy_find_sentence(full_words_str, clean_sentence, sentence_start_pos)
            if fuzzy_match:
                start_pos, end_pos, matched_text, score = fuzzy_match
                start_word_idx = position_to_word_idx[start_pos]
                end_word_idx = position_to_word_idx[end_pos - 1]
                print(
                    f"\n⚠️ Fuzzy timestamp match used for sentence: {sentence}\n"
                    f"Similarity: {score:.3f}\n"
                    f"Matched ASR text: {matched_text}"
                )
                timestamp = (
                    float(df_words['start'][start_word_idx]),
                    float(df_words['end'][end_word_idx])
                )
                if pending_empty_rows:
                    timestamp = _fill_pending_empty_timestamps(time_stamp_list, pending_empty_rows, timestamp)
                    pending_empty_rows = []
                time_stamp_list.append(timestamp)
                current_pos = end_pos
                continue

            fallback = None
            if fallback_timestamps is not None and idx < len(fallback_timestamps):
                fallback = fallback_timestamps[idx]
            if fallback is not None:
                print(
                    f"\n⚠️ ASR text missing; using the reviewed timestamp for sentence: {sentence}\n"
                    f"Timestamp: {fallback[0]:.3f}s --> {fallback[1]:.3f}s"
                )
                if pending_empty_rows:
                    fallback = _fill_pending_empty_timestamps(
                        time_stamp_list, pending_empty_rows, fallback
                    )
                    pending_empty_rows = []
                time_stamp_list.append(fallback)
                # The failed exact-search loop advances to the end of the ASR text.
                # Restore the cursor so the next spoken sentence can still match.
                current_pos = sentence_start_pos
                continue

            print(f"\n⚠️ Warning: No exact match found for sentence: {sentence}")
            show_difference(clean_sentence, 
                          full_words_str[current_pos:current_pos+len(clean_sentence)])
            print("\nOriginal sentence:", df_sentences['Source'][idx])
            raise ValueError("❎ No match found for sentence.")

    if pending_empty_rows:
        fallback = next((item for item in reversed(time_stamp_list) if item), (0.0, 0.01))
        _fill_pending_empty_timestamps(time_stamp_list, pending_empty_rows, fallback)
    
    return time_stamp_list

def _fill_pending_empty_timestamps(time_stamp_list, pending_empty_rows, next_timestamp):
    start, end = next_timestamp
    total_parts = len(pending_empty_rows) + 1
    duration = max(end - start, 0.01)
    step = duration / total_parts

    for offset, row_index in enumerate(pending_empty_rows):
        part_start = start + step * offset
        part_end = start + step * (offset + 1)
        time_stamp_list[row_index] = (part_start, part_end)

    return (start + step * len(pending_empty_rows), end)

def align_timestamp(
    df_text,
    df_translate,
    subtitle_output_configs: list,
    output_dir: str,
    for_display: bool = True,
    fallback_timestamps=None,
):
    """Align timestamps and add a new timestamp column to df_translate"""
    df_trans_time = df_translate.copy()

    # Assign an ID to each word in df_text['text'] and create a new DataFrame
    words = df_text['text'].str.split(expand=True).stack().reset_index(level=1, drop=True).reset_index()
    words.columns = ['id', 'word']
    words['id'] = words['id'].astype(int)

    # Process timestamps ⏰
    time_stamp_list = get_sentence_timestamps(
        df_text, df_translate, fallback_timestamps=fallback_timestamps
    )
    df_trans_time['timestamp'] = time_stamp_list
    df_trans_time['duration'] = df_trans_time['timestamp'].apply(lambda x: x[1] - x[0])

    # Remove gaps 🕳️
    for i in range(len(df_trans_time)-1):
        delta_time = df_trans_time.loc[i+1, 'timestamp'][0] - df_trans_time.loc[i, 'timestamp'][1]
        if 0 < delta_time < 1:
            df_trans_time.at[i, 'timestamp'] = (df_trans_time.loc[i, 'timestamp'][0], df_trans_time.loc[i+1, 'timestamp'][0])

    # Convert start and end timestamps to SRT format
    df_trans_time['timestamp'] = df_trans_time['timestamp'].apply(lambda x: convert_to_srt_format(x[0], x[1]))

    # Polish subtitles: replace punctuation in Translation if for_display
    if for_display:
        df_trans_time['Translation'] = df_trans_time['Translation'].apply(lambda x: re.sub(r'[，。]', ' ', safe_text(x)).strip())

    # Output subtitles 📜
    def generate_subtitle_string(df, columns):
        return ''.join([f"{i+1}\n{row['timestamp']}\n{safe_text(row[columns[0]]).strip()}\n{safe_text(row[columns[1]]).strip() if len(columns) > 1 else ''}\n\n" for i, row in df.iterrows()]).strip()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for filename, columns in subtitle_output_configs:
            subtitle_str = generate_subtitle_string(df_trans_time, columns)
            with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                f.write(subtitle_str)
    
    return df_trans_time

# ✨ Beautify the translation
def clean_translation(x):
    if pd.isna(x):
        return ''
    cleaned = str(x).strip('。').strip('，')
    return autocorrect.format(cleaned)

def _parse_srt_timestamp(value):
    """Return an SRT timestamp cell as a pair of seconds."""
    if pd.isna(value):
        return None
    match = re.fullmatch(
        r'\s*(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*'
        r'(\d+):(\d+):(\d+)[,.](\d+)\s*',
        str(value),
    )
    if not match:
        return None
    values = [int(item) for item in match.groups()]
    start = values[0] * 3600 + values[1] * 60 + values[2] + values[3] / 1000
    end = values[4] * 3600 + values[5] * 60 + values[6] + values[7] / 1000
    return (start, end) if end > start else None

def build_reviewed_timestamp_fallback(df_sentences, df_reviewed):
    """Map split subtitle rows back to reviewed rows and divide their time ranges."""
    if 'timestamp' not in df_reviewed.columns:
        return [None] * len(df_sentences)

    fallbacks = [None] * len(df_sentences)
    split_row = 0
    for _, reviewed_row in df_reviewed.iterrows():
        reviewed_source = remove_punctuation(
            safe_text(reviewed_row.get('Source')).lower()
        ).replace(' ', '')
        if not reviewed_source:
            continue

        group = []
        combined = ''
        while split_row < len(df_sentences) and len(combined) < len(reviewed_source):
            part = remove_punctuation(
                safe_text(df_sentences.iloc[split_row]['Source']).lower()
            ).replace(' ', '')
            group.append((split_row, max(len(part), 1)))
            combined += part
            split_row += 1

        # Only attach a reviewed time when the split rows reconstruct the source.
        # This prevents a timestamp from being assigned to an unrelated later row.
        if combined != reviewed_source:
            continue
        timestamp = _parse_srt_timestamp(reviewed_row.get('timestamp'))
        if timestamp is None:
            continue

        start, end = timestamp
        total_weight = sum(weight for _, weight in group)
        elapsed_weight = 0
        for row_index, weight in group:
            part_start = start + (end - start) * elapsed_weight / total_weight
            elapsed_weight += weight
            part_end = start + (end - start) * elapsed_weight / total_weight
            fallbacks[row_index] = (part_start, part_end)

    return fallbacks

def align_timestamp_main():
    df_text = pd.read_excel(_2_CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].apply(lambda x: safe_text(x).strip('"').strip())
    df_translate = pd.read_excel(_5_SPLIT_SUB)
    df_translate['Source'] = df_translate['Source'].apply(safe_text)
    df_translate['Translation'] = df_translate['Translation'].apply(clean_translation)
    df_reviewed = pd.read_excel(_4_2_TRANSLATION)
    fallback_timestamps = build_reviewed_timestamp_fallback(df_translate, df_reviewed)

    align_timestamp(
        df_text,
        df_translate,
        SUBTITLE_OUTPUT_CONFIGS,
        _OUTPUT_DIR,
        fallback_timestamps=fallback_timestamps,
    )
    console.print(Panel("[bold green]🎉📝 Subtitles generation completed! Please check in the `output` folder 👀[/bold green]"))

    # for audio
    df_translate_for_audio = pd.read_excel(_5_REMERGED) # use remerged file to avoid unmatched lines when dubbing
    df_translate_for_audio['Source'] = df_translate_for_audio['Source'].apply(safe_text)
    df_translate_for_audio['Translation'] = df_translate_for_audio['Translation'].apply(clean_translation)
    
    audio_fallback_timestamps = build_reviewed_timestamp_fallback(
        df_translate_for_audio, df_reviewed
    )
    align_timestamp(
        df_text,
        df_translate_for_audio,
        AUDIO_SUBTITLE_OUTPUT_CONFIGS,
        _AUDIO_DIR,
        fallback_timestamps=audio_fallback_timestamps,
    )
    console.print(Panel(f"[bold green]🎉📝 Audio subtitles generation completed! Please check in the `{_AUDIO_DIR}` folder 👀[/bold green]"))
    

if __name__ == '__main__':
    align_timestamp_main()
