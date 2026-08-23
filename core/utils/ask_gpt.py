import os
import json
import math
from datetime import datetime
from threading import Lock
import json_repair
from openai import OpenAI
from core.utils.config_utils import load_key
from rich import print as rprint
from core.utils.decorator import except_handler

# ------------
# cache gpt response
# ------------

LOCK = Lock()
GPT_LOG_FOLDER = 'output/gpt_log'


def _load_optional_key(key, default=None):
    try:
        return load_key(key)
    except KeyError:
        return default

def _to_plain_dict(value):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if hasattr(value, key):
            result[key] = getattr(value, key)
    return result or None

def _estimate_tokens(text):
    # Model-agnostic fallback for local providers that omit usage metadata.
    return math.ceil(len(str(text or "")) / 4)

def _usage_event(model, prompt, resp_content, resp_type, usage, log_title):
    usage = _to_plain_dict(usage)
    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "log_title": log_title,
        "model": model,
        "resp_type": resp_type,
        "prompt_chars": len(prompt or ""),
        "response_chars": len(resp_content or ""),
        "estimated_prompt_tokens": _estimate_tokens(prompt),
        "estimated_completion_tokens": _estimate_tokens(resp_content),
        "usage": usage,
    }
    event["estimated_total_tokens"] = event["estimated_prompt_tokens"] + event["estimated_completion_tokens"]
    if usage:
        event["prompt_tokens"] = usage.get("prompt_tokens")
        event["completion_tokens"] = usage.get("completion_tokens")
        event["total_tokens"] = usage.get("total_tokens")
    return event

def _write_usage_logs(event):
    events_file = os.path.join(GPT_LOG_FOLDER, "usage_events.json")
    summary_file = os.path.join(GPT_LOG_FOLDER, "usage_summary.json")

    events = []
    if os.path.exists(events_file):
        with open(events_file, 'r', encoding='utf-8') as f:
            events = json.load(f)
    events.append(event)
    with open(events_file, 'w', encoding='utf-8') as f:
        json.dump(events, f, ensure_ascii=False, indent=4)

    summary = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "requests": len(events),
        "estimated_prompt_tokens": sum(item.get("estimated_prompt_tokens", 0) for item in events),
        "estimated_completion_tokens": sum(item.get("estimated_completion_tokens", 0) for item in events),
        "estimated_total_tokens": sum(item.get("estimated_total_tokens", 0) for item in events),
        "exact_prompt_tokens": sum(item.get("prompt_tokens") or 0 for item in events),
        "exact_completion_tokens": sum(item.get("completion_tokens") or 0 for item in events),
        "exact_total_tokens": sum(item.get("total_tokens") or 0 for item in events),
        "exact_usage_entries": sum(1 for item in events if item.get("total_tokens") is not None),
        "by_log_title": {},
    }
    for item in events:
        bucket = summary["by_log_title"].setdefault(
            item["log_title"],
            {
                "requests": 0,
                "estimated_prompt_tokens": 0,
                "estimated_completion_tokens": 0,
                "estimated_total_tokens": 0,
                "exact_total_tokens": 0,
            },
        )
        bucket["requests"] += 1
        bucket["estimated_prompt_tokens"] += item.get("estimated_prompt_tokens", 0)
        bucket["estimated_completion_tokens"] += item.get("estimated_completion_tokens", 0)
        bucket["estimated_total_tokens"] += item.get("estimated_total_tokens", 0)
        bucket["exact_total_tokens"] += item.get("total_tokens") or 0

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

def _save_cache(model, prompt, resp_content, resp_type, resp, message=None, log_title="default", usage=None, base_url=None):
    with LOCK:
        logs = []
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        usage = _to_plain_dict(usage)
        logs.append({"model": model, "base_url": base_url, "prompt": prompt, "resp_content": resp_content, "resp_type": resp_type, "resp": resp, "message": message, "usage": usage})
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)
        _write_usage_logs(_usage_event(model, prompt, resp_content, resp_type, usage, log_title))

def _load_cache(prompt, resp_type, log_title, model=None, base_url=None):
    with LOCK:
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        if os.path.exists(file):
            with open(file, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    if item["prompt"] == prompt and item["resp_type"] == resp_type:
                        if model and item.get("model") and item.get("model") != model:
                            continue
                        if base_url and item.get("base_url") and item.get("base_url") != base_url:
                            continue
                        return item["resp"]
        return False

# ------------
# ask gpt once
# ------------

@except_handler("GPT request failed", retry=5)
def ask_gpt(
    prompt,
    resp_type=None,
    valid_def=None,
    log_title="default",
    api_config=None,
    use_cache=True,
    attempt_tracker=None,
):
    api_config = api_config or {}
    api_key = api_config.get("key") or load_key("api.key")
    if not api_key:
        raise ValueError("API key is not set")
    model = api_config.get("model") or load_key("api.model")
    base_url = api_config.get("base_url") or load_key("api.base_url")
    if 'ark' in base_url:
        base_url = "https://ark.cn-beijing.volces.com/api/v3" # huoshan base url
    elif 'v1' not in base_url:
        base_url = base_url.strip('/') + '/v1'
    # check cache after resolving the effective model and base URL
    if use_cache:
        cached = _load_cache(prompt, resp_type, log_title, model=model, base_url=base_url)
        if cached:
            if attempt_tracker is not None:
                attempt_tracker["cache_hits"] = attempt_tracker.get("cache_hits", 0) + 1
            rprint("use cache response")
            return cached

    client = OpenAI(api_key=api_key, base_url=base_url)
    llm_support_json = api_config.get("llm_support_json")
    if llm_support_json is None:
        llm_support_json = load_key("api.llm_support_json")
    response_format = {"type": "json_object"} if resp_type == "json" and llm_support_json else None
    reasoning_config = api_config.get("reasoning")
    if reasoning_config is None:
        reasoning_config = _load_optional_key("api.reasoning")
    extra_body = dict(api_config.get("extra_body") or {})
    if reasoning_config:
        extra_body["reasoning"] = dict(reasoning_config)

    messages = [{"role": "user", "content": prompt}]

    params = dict(
        model=model,
        messages=messages,
        response_format=response_format,
        timeout=300
    )
    if extra_body:
        params["extra_body"] = extra_body
    if attempt_tracker is not None:
        attempt_tracker["requests"] = attempt_tracker.get("requests", 0) + 1
    resp_raw = client.chat.completions.create(**params)

    # process and return full result
    resp_content = resp_raw.choices[0].message.content
    usage = _to_plain_dict(getattr(resp_raw, "usage", None))
    if resp_type == "json":
        resp = json_repair.loads(resp_content)
    else:
        resp = resp_content
    
    # check if the response format is valid
    if valid_def:
        valid_resp = valid_def(resp)
        if valid_resp['status'] != 'success':
            if attempt_tracker is not None:
                attempt_tracker["validation_errors"] = attempt_tracker.get("validation_errors", 0) + 1
            _save_cache(model, prompt, resp_content, resp_type, resp, log_title="error", message=valid_resp['message'], usage=usage, base_url=base_url)
            raise ValueError(f"❎ API response error: {valid_resp['message']}")

    _save_cache(model, prompt, resp_content, resp_type, resp, log_title=log_title, usage=usage, base_url=base_url)
    return resp


if __name__ == '__main__':
    from rich import print as rprint
    
    result = ask_gpt("""test respond ```json\n{\"code\": 200, \"message\": \"success\"}\n```""", resp_type="json")
    rprint(f"Test json output result: {result}")
