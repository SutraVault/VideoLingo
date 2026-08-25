import concurrent.futures
from difflib import SequenceMatcher
import math
from core.prompts import get_split_prompt
from core.utils.llm_stage_utils import stage_api_config
from core.spacy_utils.load_nlp_model import init_nlp
from core.utils import *
from rich.console import Console
from rich.table import Table
from core.utils.models import _3_1_SPLIT_BY_NLP, _3_2_SPLIT_BY_MEANING
console = Console()

def tokenize_sentence(sentence, nlp):
    doc = nlp(sentence)
    return [token.text for token in doc]

def find_split_positions(original, modified):
    split_positions = []
    parts = modified.split('[br]')
    start = 0
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)

    for i in range(len(parts) - 1):
        max_similarity = 0
        best_split = None

        for j in range(start, len(original)):
            original_left = original[start:j]
            modified_left = joiner.join(parts[i].split())

            left_similarity = SequenceMatcher(None, original_left, modified_left).ratio()

            if left_similarity > max_similarity:
                max_similarity = left_similarity
                best_split = j

        if max_similarity < 0.9:
            console.print(f"[yellow]Warning: low similarity found at the best split point: {max_similarity}[/yellow]")
        if best_split is not None:
            split_positions.append(best_split)
            start = best_split
        else:
            console.print(f"[yellow]Warning: Unable to find a suitable split point for the {i+1}th part.[/yellow]")

    return split_positions

def normalize_split_response(response_data):
    """Return the split response dict when providers wrap JSON in a list."""
    if isinstance(response_data, list):
        merged = {}
        for item in response_data:
            if isinstance(item, dict):
                merged.update(item)
        if merged:
            response_data = merged
        else:
            return None

    if not isinstance(response_data, dict):
        return None

    return response_data

def get_split_key(response_data):
    choice = str(response_data.get("choice", "")).strip()
    if choice.startswith("split"):
        return choice
    if choice:
        return f"split{choice}"

    for key in ("split", "split1", "split2", "split_1", "split_2"):
        if key in response_data and "[br]" in str(response_data[key]):
            return key

    for key, value in response_data.items():
        if str(key).startswith("split") and "[br]" in str(value):
            return key

    return None


def fallback_split_sentence(sentence, num_parts):
    """Split locally when the LLM fails to return usable JSON."""
    text = str(sentence).strip()
    if num_parts <= 1 or not text:
        return text

    if " " not in text:
        chunk_size = max(1, math.ceil(len(text) / num_parts))
        return "\n".join(
            text[index:index + chunk_size].strip()
            for index in range(0, len(text), chunk_size)
            if text[index:index + chunk_size].strip()
        )

    words = text.split()
    chunk_size = max(1, math.ceil(len(words) / num_parts))
    parts = [
        " ".join(words[index:index + chunk_size]).strip()
        for index in range(0, len(words), chunk_size)
    ]
    return "\n".join(part for part in parts if part)


def split_sentence(sentence, num_parts, word_limit=20, index=-1, retry_attempt=0, attempt_tracker=None):
    """Split a long sentence using GPT and return the result as a string."""
    split_prompt = get_split_prompt(sentence, num_parts, word_limit)
    def valid_split(response_data):
        response_data = normalize_split_response(response_data)
        if response_data is None:
            return {"status": "error", "message": "Response must be a JSON object"}

        split_key = get_split_key(response_data)
        if split_key not in response_data:
            return {"status": "error", "message": "Missing required key: `split`"}
        split_text = str(response_data[split_key])
        parts = [part.strip() for part in split_text.split("[br]")]
        if len(parts) != num_parts or any(not part for part in parts):
            return {"status": "error", "message": f"Expected exactly {num_parts} non-empty parts"}
        if any(len(part.split()) > word_limit for part in parts):
            return {"status": "error", "message": f"A split part exceeds {word_limit} words"}
        return {"status": "success", "message": "Split completed"}
    
    try:
        response_data = ask_gpt(
            split_prompt + " " * retry_attempt,
            resp_type='json',
            valid_def=valid_split,
            log_title='split_by_meaning',
            attempt_tracker=attempt_tracker,
            api_config=stage_api_config("split"),
        )
        response_data = normalize_split_response(response_data)
        split_key = get_split_key(response_data)
        best_split = response_data[split_key]
        split_points = find_split_positions(sentence, best_split)
    except Exception as exc:
        console.print(
            "[yellow]Warning: LLM sentence split failed after retries. "
            f"Using local fallback split. Details: {exc}[/yellow]"
        )
        best_split = fallback_split_sentence(sentence, num_parts)
        split_points = []

    # split the sentence based on the split points
    for i, split_point in enumerate(split_points):
        if i == 0:
            best_split = sentence[:split_point] + '\n' + sentence[split_point:]
        else:
            parts = best_split.split('\n')
            last_part = parts[-1]
            parts[-1] = last_part[:split_point - split_points[i-1]] + '\n' + last_part[split_point - split_points[i-1]:]
            best_split = '\n'.join(parts)
    if index != -1:
        console.print(f'[green]✅ Sentence {index} has been successfully split[/green]')
    table = Table(title="")
    table.add_column("Type", style="cyan")
    table.add_column("Sentence")
    table.add_row("Original", sentence, style="yellow")
    table.add_row("Split", best_split.replace('\n', ' ||'), style="yellow")
    console.print(table)
    
    return best_split

def parallel_split_sentences(sentences, max_length, max_workers, nlp, retry_attempt=0):
    """Split sentences in parallel using a thread pool."""
    new_sentences = [None] * len(sentences)
    futures = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, sentence in enumerate(sentences):
            # Use tokenizer to split the sentence
            tokens = tokenize_sentence(sentence, nlp)
            # print("Tokenization result:", tokens)
            num_parts = math.ceil(len(tokens) / max_length)
            if len(tokens) > max_length:
                future = executor.submit(split_sentence, sentence, num_parts, max_length, index=index, retry_attempt=retry_attempt)
                futures.append((future, index, num_parts, sentence))
            else:
                new_sentences[index] = [sentence]

        for future, index, num_parts, sentence in futures:
            split_result = future.result()
            if split_result:
                split_lines = split_result.strip().split('\n')
                new_sentences[index] = [line.strip() for line in split_lines]
            else:
                new_sentences[index] = [sentence]

    return [sentence for sublist in new_sentences for sentence in sublist]

@check_file_exists(_3_2_SPLIT_BY_MEANING)
def split_sentences_by_meaning():
    """The main function to split sentences by meaning."""
    # read input sentences
    with open(_3_1_SPLIT_BY_NLP, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]

    nlp = init_nlp()
    # 🔄 process sentences multiple times to ensure all are split
    for retry_attempt in range(3):
        sentences = parallel_split_sentences(sentences, max_length=load_key("max_split_length"), max_workers=load_key("max_workers"), nlp=nlp, retry_attempt=retry_attempt)

    # 💾 save results
    with open(_3_2_SPLIT_BY_MEANING, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sentences))
    console.print('[green]✅ All sentences have been successfully split![/green]')

if __name__ == '__main__':
    # print(split_sentence('Which makes no sense to the... average guy who always pushes the character creation slider all the way to the right.', 2, 22))
    split_sentences_by_meaning()
