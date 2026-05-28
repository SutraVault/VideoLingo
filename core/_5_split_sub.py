import pandas as pd
from typing import List, Tuple
import concurrent.futures

from core._3_2_split_meaning import split_sentence
from core.prompts import get_align_prompt
from core.utils.excel_utils import read_excel_with_aliases
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from core.utils import *
from core.utils.models import *
console = Console()

# ! You can modify your own weights here
# Chinese and Japanese 2.5 characters, Korean 2 characters, Thai 1.5 characters, full-width symbols 2 characters, other English-based and half-width symbols 1 character
def calc_len(text: str) -> float:
    text = str(text) # force convert
    def char_weight(char):
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:  # Chinese and Japanese
            return 1.75
        elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:  # Korean
            return 1.5
        elif 0x0E00 <= code <= 0x0E7F:  # Thai
            return 1
        elif 0xFF01 <= code <= 0xFF5E:  # full-width symbols
            return 1.75
        else:  # other characters (e.g. English and half-width symbols)
            return 1

    return sum(char_weight(char) for char in text)

def split_text_by_weights(text: str, weights: List[float]) -> List[str]:
    """Fallback splitter used when the LLM returns empty alignment parts."""
    text = str(text).strip()
    if not weights:
        return [text] if text else []

    if not text:
        return [""] * len(weights)

    total_weight = sum(max(weight, 1) for weight in weights)
    target_lengths = [max(1, round(len(text) * max(weight, 1) / total_weight)) for weight in weights]

    parts = []
    start = 0
    for i, target_length in enumerate(target_lengths):
        remaining_parts = len(weights) - i
        if remaining_parts == 1:
            parts.append(text[start:].strip())
            break

        remaining_chars = len(text) - start
        cut = start + min(target_length, remaining_chars - (remaining_parts - 1))
        cut = max(start + 1, cut)

        search_start = max(start + 1, cut - 8)
        search_end = min(len(text) - (remaining_parts - 1), cut + 8)
        candidates = [
            pos + 1
            for pos in range(search_start - 1, search_end)
            if text[pos] in " ,，.。;；:：!?！？、"
        ]
        if candidates:
            cut = min(candidates, key=lambda pos: abs(pos - cut))

        parts.append(text[start:cut].strip())
        start = cut

    return [part if part else text[:1] for part in parts]

def normalize_align_data(response_data, tr_sub: str, src_parts: List[str]) -> List[str]:
    align_data = response_data.get('align', [])
    tr_parts = []
    for i, item in enumerate(align_data):
        key = f'target_part_{i+1}'
        value = item.get(key, item.get('target_part', item.get('target', '')))
        tr_parts.append(str(value).strip())

    if len(tr_parts) == len(src_parts) and all(tr_parts):
        return tr_parts

    console.print(
        "[yellow]Warning: LLM subtitle alignment returned empty or incomplete target parts. "
        "Using local proportional fallback split.[/yellow]"
    )
    return split_text_by_weights(tr_sub, [calc_len(part) for part in src_parts])

def align_subs(src_sub: str, tr_sub: str, src_part: str) -> Tuple[List[str], List[str], str]:
    align_prompt = get_align_prompt(src_sub, tr_sub, src_part)
    src_parts = src_part.split('\n')
    expected_parts = len(src_parts)
    
    def valid_align(response_data):
        if 'align' not in response_data:
            return {"status": "error", "message": "Missing required key: `align`"}
        if len(response_data['align']) != expected_parts:
            return {
                "status": "error",
                "message": f"Align returned {len(response_data['align'])} parts, expected {expected_parts}",
            }
        if not all(isinstance(item, dict) for item in response_data['align']):
            return {"status": "error", "message": "`align` must be a list of JSON objects"}
        return {"status": "success", "message": "Align completed"}

    parsed = ask_gpt(align_prompt, resp_type='json', valid_def=valid_align, log_title='align_subs_v2')
    tr_parts = normalize_align_data(parsed, tr_sub, src_parts)
    
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)
    tr_remerged = joiner.join(tr_parts)
    
    table = Table(title="🔗 Aligned parts")
    table.add_column("Language", style="cyan")
    table.add_column("Parts", style="magenta")
    table.add_row("SRC_LANG", "\n".join(src_parts))
    table.add_row("TARGET_LANG", "\n".join(tr_parts))
    console.print(table)
    
    return src_parts, tr_parts, tr_remerged

def split_align_subs(src_lines: List[str], tr_lines: List[str]):
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    remerged_tr_lines = tr_lines.copy()
    
    to_split = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        src, tr = str(src), str(tr)
        if len(src) > MAX_SUB_LENGTH or calc_len(tr) * TARGET_SUB_MULTIPLIER > MAX_SUB_LENGTH:
            to_split.append(i)
            table = Table(title=f"📏 Line {i} needs to be split")
            table.add_column("Type", style="cyan")
            table.add_column("Content", style="magenta")
            table.add_row("Source Line", src)
            table.add_row("Target Line", tr)
            console.print(table)
    
    @except_handler("Error in split_align_subs")
    def process(i):
        split_src = split_sentence(src_lines[i], num_parts=2).strip()
        src_parts, tr_parts, tr_remerged = align_subs(src_lines[i], tr_lines[i], split_src)
        src_lines[i] = src_parts
        tr_lines[i] = tr_parts
        remerged_tr_lines[i] = tr_remerged
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
        list(executor.map(process, to_split))
    
    # Flatten `src_lines` and `tr_lines`
    src_lines = [item for sublist in src_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    tr_lines = [item for sublist in tr_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    if len(src_lines) != len(tr_lines):
        raise ValueError(
            f"Subtitle split alignment produced mismatched lengths: "
            f"{len(src_lines)} source rows vs {len(tr_lines)} translation rows"
        )
    
    return src_lines, tr_lines, remerged_tr_lines

def split_for_sub_main():
    console.print("[bold green]🚀 Start splitting subtitles...[/bold green]")
    
    df = read_excel_with_aliases(_4_2_TRANSLATION, required_columns=['Source', 'Translation'])
    src = df['Source'].tolist()
    trans = df['Translation'].tolist()
    
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    
    for attempt in range(3):  # 多次切割
        console.print(Panel(f"🔄 Split attempt {attempt + 1}", expand=False))
        split_src, split_trans, remerged = split_align_subs(src.copy(), trans.copy())
        
        # 检查是否所有字幕都符合长度要求
        if all(len(src) <= MAX_SUB_LENGTH for src in split_src) and \
           all(calc_len(tr) * TARGET_SUB_MULTIPLIER <= MAX_SUB_LENGTH for tr in split_trans):
            break
        
        # 更新源数据继续下一轮分割
        src, trans = split_src, split_trans

    # 确保二者有相同的长度，防止报错
    if len(src) > len(remerged):
        remerged += [None] * (len(src) - len(remerged))
    elif len(remerged) > len(src):
        src += [None] * (len(remerged) - len(src))
    
    if len(split_src) != len(split_trans):
        raise ValueError(
            f"Cannot write subtitle split output with mismatched lengths: "
            f"{len(split_src)} source rows vs {len(split_trans)} translation rows"
        )

    pd.DataFrame({'Source': split_src, 'Translation': split_trans}).to_excel(_5_SPLIT_SUB, index=False)
    pd.DataFrame({'Source': src, 'Translation': remerged}).to_excel(_5_REMERGED, index=False)

if __name__ == '__main__':
    split_for_sub_main()
