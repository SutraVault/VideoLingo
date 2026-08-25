from core.prompts import generate_shared_prompt, get_prompt_faithfulness, get_prompt_expressiveness
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich import box
from core.utils import *
from core.utils.llm_stage_utils import stage_api_config
console = Console()

def _normalize_item(item, required_sub_keys):
    if not isinstance(item, dict):
        return {required_sub_keys[0]: str(item)}
    normalized = dict(item)
    for sub_key in required_sub_keys:
        if sub_key in normalized:
            continue
        for alias in ("translation", "translated", "text", "result", "content", "subtitle"):
            if alias in normalized:
                normalized[sub_key] = normalized[alias]
                break
    return normalized

def normalize_translate_result(result, required_sub_keys: list):
    if isinstance(result, dict):
        for wrapper_key in ("translations", "translation", "results", "items", "data"):
            nested = result.get(wrapper_key)
            if isinstance(nested, (dict, list)):
                result = nested
                break

    normalized = {}
    if isinstance(result, list):
        for index, item in enumerate(result, 1):
            key = str(index)
            normalized[key] = _normalize_item(item, required_sub_keys)
        return normalized

    if isinstance(result, dict):
        for key, item in result.items():
            normalized_key = str(key).strip()
            normalized[normalized_key] = _normalize_item(item, required_sub_keys)
        return normalized

    return {}

def valid_translate_result(result, required_keys: list, required_sub_keys: list):
    result = normalize_translate_result(result, required_sub_keys)
    # Check for the required key
    if not all(key in result for key in required_keys):
        return {"status": "error", "message": f"Missing required key(s): {', '.join(set(required_keys) - set(result.keys()))}"}
    
    # Check for required sub-keys in all items
    for key in result:
        if not all(sub_key in result[key] for sub_key in required_sub_keys):
            return {"status": "error", "message": f"Missing required sub-key(s) in item {key}: {', '.join(set(required_sub_keys) - set(result[key].keys()))}"}

    return {"status": "success", "message": "Translation completed"}

def translate_lines(lines, previous_content_prompt, after_cotent_prompt, things_to_note_prompt, summary_prompt, index = 0):
    shared_prompt = generate_shared_prompt(previous_content_prompt, after_cotent_prompt, summary_prompt, things_to_note_prompt)
    source_lines = lines.split('\n')

    # Retry translation if the length of the original text and the translated text are not the same, or if the specified key is missing
    def retry_translation(prompt, length, step_name):
        api_config = stage_api_config("translate")
        def valid_faith(response_data):
            return valid_translate_result(response_data, [str(i) for i in range(1, length+1)], ['direct'])
        def valid_express(response_data):
            return valid_translate_result(response_data, [str(i) for i in range(1, length+1)], ['free'])
        for retry in range(3):
            required_sub_keys = ['direct'] if step_name == 'faithfulness' else ['free']
            validator = valid_faith if step_name == 'faithfulness' else valid_express
            if step_name == 'faithfulness':
                result = ask_gpt(prompt+retry* " ", resp_type='json', log_title=f'translate_{step_name}', api_config=api_config)
            elif step_name == 'expressiveness':
                result = ask_gpt(prompt+retry* " ", resp_type='json', log_title=f'translate_{step_name}', api_config=api_config)
            result = normalize_translate_result(result, required_sub_keys)
            valid_resp = validator(result)
            if valid_resp['status'] == 'success' and len(lines.split('\n')) == len(result):
                return result
            if retry != 2:
                console.print(f"[yellow]⚠️ {step_name.capitalize()} translation of block {index} failed: {valid_resp['message']}. Retry...[/yellow]")
        raise ValueError(f'[red]❌ {step_name.capitalize()} translation of block {index} failed after 3 retries. Please check `output/gpt_log/error.json` for more details.[/red]')

    ## Step 1: Faithful to the Original Text
    prompt1 = get_prompt_faithfulness(lines, shared_prompt)
    faith_result = retry_translation(prompt1, len(lines.split('\n')), 'faithfulness')

    for i in faith_result:
        if "origin" not in faith_result[i]:
            line_index = int(i) - 1
            if 0 <= line_index < len(source_lines):
                faith_result[i]["origin"] = source_lines[line_index]
        faith_result[i]["direct"] = faith_result[i]["direct"].replace('\n', ' ')

    # If reflect_translate is False or not set, use faithful translation directly
    reflect_translate = load_key('reflect_translate')
    if not reflect_translate:
        # If reflect_translate is False or not set, use faithful translation directly
        translate_result = "\n".join([faith_result[str(i)]["direct"].strip() for i in range(1, len(source_lines) + 1)])
        
        table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
        table.add_column("Translations", style="bold")
        for i, key in enumerate(str(i) for i in range(1, len(source_lines) + 1)):
            table.add_row(f"[cyan]Origin:  {faith_result[key]['origin']}[/cyan]")
            table.add_row(f"[magenta]Direct:  {faith_result[key]['direct']}[/magenta]")
            if i < len(faith_result) - 1:
                table.add_row("[yellow]" + "-" * 50 + "[/yellow]")
        
        console.print(table)
        return translate_result, lines

    ## Step 2: Express Smoothly  
    prompt2 = get_prompt_expressiveness(faith_result, lines, shared_prompt)
    express_result = retry_translation(prompt2, len(lines.split('\n')), 'expressiveness')

    table = Table(title="Translation Results", show_header=False, box=box.ROUNDED)
    table.add_column("Translations", style="bold")
    for i, key in enumerate(str(i) for i in range(1, len(source_lines) + 1)):
        table.add_row(f"[cyan]Origin:  {faith_result[key]['origin']}[/cyan]")
        table.add_row(f"[magenta]Direct:  {faith_result[key]['direct']}[/magenta]")
        table.add_row(f"[green]Free:    {express_result[key]['free']}[/green]")
        if i < len(express_result) - 1:
            table.add_row("[yellow]" + "-" * 50 + "[/yellow]")

    console.print(table)

    translate_result = "\n".join([express_result[str(i)]["free"].replace('\n', ' ').strip() for i in range(1, len(source_lines) + 1)])

    if len(lines.split('\n')) != len(translate_result.split('\n')):
        console.print(Panel(f'[red]❌ Translation of block {index} failed, Length Mismatch, Please check `output/gpt_log/translate_expressiveness.json`[/red]'))
        raise ValueError(f'Origin ···{lines}···,\nbut got ···{translate_result}···')

    return translate_result, lines


if __name__ == '__main__':
    # test e.g.
    lines = '''All of you know Andrew Ng as a famous computer science professor at Stanford.
He was really early on in the development of neural networks with GPUs.
Of course, a creator of Coursera and popular courses like deeplearning.ai.
Also the founder and creator and early lead of Google Brain.'''
    previous_content_prompt = None
    after_cotent_prompt = None
    things_to_note_prompt = None
    summary_prompt = None
    translate_lines(lines, previous_content_prompt, after_cotent_prompt, things_to_note_prompt, summary_prompt)
