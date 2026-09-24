import streamlit as st
import requests
import html
from translations.translations import translate as t
from translations.translations import DISPLAY_LANGUAGES
from core.utils import *


def config_input(label, key, help=None, placeholder=None):
    """Generic config input handler"""
    val = st.text_input(label, value=load_key(key), help=help, placeholder=placeholder)
    if val != load_key(key):
        update_key(key, val)
    return val


def _fetch_model_list(base_url, api_key):
    """Fetch available models from OpenAI-compatible /v1/models endpoint."""
    if not api_key or not base_url:
        return []
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    url += "/models"
    try:
        resp = requests.get(
            url, headers={"Authorization": f"Bearer {api_key}"}, timeout=10
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return sorted([m["id"] for m in data if "id" in m])
    except Exception:
        return []


def _search_models(search_term, **kwargs):
    """Search function for st_searchbox — returns models matching the search term."""
    models = st.session_state.get("_model_list", [])
    if not search_term:
        return models if models else []
    term = search_term.lower()
    matched = [m for m in models if term in m.lower()]
    # Always include the raw input as an option so users can type custom model names
    if search_term not in matched:
        matched.insert(0, search_term)
    return matched


def _stage_model_input(label, key):
    current = str(load_key(key))
    models = st.session_state.get("_model_list", [])
    options = list(dict.fromkeys([current] + models))
    if models:
        selected = st.selectbox(label, options=options, index=0, key=f"ui_{key}")
    else:
        selected = st.text_input(label, value=current, key=f"ui_{key}")
    if selected != current:
        update_key(key, selected)
    return selected


def _reasoning_input(label, key, allow_off=True):
    options = ["auto"] + (["off"] if allow_off else []) + ["low", "medium", "high", "xhigh"]
    current = str(load_key(key))
    selected = st.selectbox(
        label,
        options=options,
        index=options.index(current) if current in options else 0,
        help="Auto sends no override; support for Off and effort levels depends on the selected model/provider.",
        key=f"ui_{key}",
    )
    if selected != current:
        update_key(key, selected)
        st.rerun()
    return selected

def _watermark_preview_style(position):
    positions = {
        "top_left": "top: 12px; left: 12px;",
        "top_right": "top: 12px; right: 12px;",
        "bottom_left": "bottom: 12px; left: 12px;",
        "bottom_right": "bottom: 12px; right: 12px;",
        "center": "top: 50%; left: 50%; transform: translate(-50%, -50%);",
    }
    return positions.get(position, positions["top_right"])

def _render_watermark_preview(text, position, opacity, font_size):
    preview_font_size = max(8, min(48, int(font_size * 0.75)))
    st.markdown(
        f"""
        <div style="
            position: relative;
            width: 100%;
            aspect-ratio: 16 / 9;
            background: #111;
            border: 1px solid #333;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                {_watermark_preview_style(position)}
                color: rgba(255, 255, 255, {opacity});
                font-size: {preview_font_size}px;
                font-family: Arial, sans-serif;
                font-weight: 600;
                text-shadow: 0 0 2px rgba(0, 0, 0, {opacity});
                white-space: nowrap;
                max-width: calc(100% - 24px);
                overflow: hidden;
                text-overflow: ellipsis;
            ">{html.escape(text) if text else ""}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_setting():
    # Widen the sidebar slightly to accommodate the model searchbox
    st.markdown(
        """<style>[data-testid="stSidebar"] {min-width: 420px; max-width: 420px;}</style>""",
        unsafe_allow_html=True,
    )

    display_language = st.selectbox(
        "Display Language 🌐",
        options=list(DISPLAY_LANGUAGES.keys()),
        index=list(DISPLAY_LANGUAGES.values()).index(load_key("display_language")),
    )
    if DISPLAY_LANGUAGES[display_language] != load_key("display_language"):
        update_key("display_language", DISPLAY_LANGUAGES[display_language])
        st.rerun()

    with st.expander(t("Youtube Settings"), expanded=False):
        config_input(
            t("Cookies Path"),
            "youtube.cookies_path",
            help=t("Path to a Netscape-format cookies.txt exported from your browser for YouTube."),
        )

    with st.expander(t("LLM Configuration"), expanded=True):
        config_input(t("API_KEY"), "api.key", placeholder=t("Enter your API key"))
        config_input(
            t("BASE_URL"),
            "api.base_url",
            help=t("Openai format, will add /v1/chat/completions automatically"),
        )

        # Try to use searchbox for model selection, fall back to text_input
        try:
            from streamlit_searchbox import st_searchbox
            from streamlit_searchbox import _list_to_options_js, _list_to_options_py

            if st.button(
                t("Fetch Model List"), key="fetch_models", width="stretch"
            ):
                with st.spinner(t("Fetching models...")):
                    models = _fetch_model_list(
                        load_key("api.base_url"), load_key("api.key")
                    )
                    st.session_state["_model_list"] = models
                    if models:
                        # Update searchbox internal state directly so dropdown shows options
                        sb_key = "model_searchbox"
                        if sb_key in st.session_state:
                            st.session_state[sb_key]["options_js"] = (
                                _list_to_options_js(models)
                            )
                            st.session_state[sb_key]["options_py"] = (
                                _list_to_options_py(models)
                            )
                        st.toast(
                            t("Fetched {n} models").replace("{n}", str(len(models))),
                            icon="✅",
                        )
                    else:
                        st.toast(
                            t(
                                "Failed to fetch models, please check API Key and Base URL"
                            ),
                            icon="❌",
                        )

            current_model = load_key("api.model")
            model_list = st.session_state.get("_model_list", None)

            sb_key = "model_searchbox"
            selected = st_searchbox(
                _search_models,
                placeholder=t("Search or enter model name"),
                default=current_model if current_model else None,
                default_searchterm=current_model if current_model else "",
                default_use_searchterm=True,
                default_options=model_list if model_list else None,
                key=sb_key,
                clear_on_submit=False,
            )
            if selected and selected != load_key("api.model"):
                update_key("api.model", selected)

            if st.button("📡 " + t("Check API"), key="api", width="stretch"):
                with st.spinner(t("Check API") + "..."):
                    is_valid = check_api()
                st.toast(
                    t("API Key is valid") if is_valid else t("API Key is invalid"),
                    icon="✅" if is_valid else "❌",
                )
        except ImportError:
            c1, c2 = st.columns([4, 1])
            with c1:
                config_input(
                    t("MODEL"),
                    "api.model",
                    help=t("click to check API validity") + " 👉",
                    placeholder=t("Search or enter model name"),
                )
            with c2:
                if st.button("📡", key="api"):
                    is_valid = check_api()
                    st.toast(
                        t("API Key is valid") if is_valid else t("API Key is invalid"),
                        icon="✅" if is_valid else "❌",
                    )
        llm_support_json = st.toggle(
            t("LLM JSON Format Support"),
            value=load_key("api.llm_support_json"),
            help=t("Enable if your LLM supports JSON mode output"),
        )
        if llm_support_json != load_key("api.llm_support_json"):
            update_key("api.llm_support_json", llm_support_json)
            st.rerun()

        staged_enabled = st.toggle(
            "Stage-specific LLM routing",
            value=load_key("llm_stages.enabled"),
            help="Choose independent models and reasoning modes for splitting, translation, hard lines, and proofreading.",
        )
        if staged_enabled != load_key("llm_stages.enabled"):
            update_key("llm_stages.enabled", staged_enabled)
            st.rerun()

        if staged_enabled:
            st.caption("Any OpenAI-compatible/OpenRouter model can be used; the displayed defaults are editable examples.")
            with st.expander("Stage-specific LLM settings", expanded=True):
                _stage_model_input("Summary and terminology model", "llm_stages.summary.model")
                _reasoning_input("Summary reasoning", "llm_stages.summary.reasoning")

                _stage_model_input("Sentence splitting model", "llm_stages.split.model")
                _reasoning_input("Sentence splitting reasoning", "llm_stages.split.reasoning")

                _stage_model_input("Translation model", "llm_stages.translate.model")
                _reasoning_input("Translation reasoning", "llm_stages.translate.reasoning")

                hard_enabled = st.checkbox(
                    "Second pass for difficult lines",
                    value=load_key("llm_stages.hard_translation.enabled"),
                )
                if hard_enabled != load_key("llm_stages.hard_translation.enabled"):
                    update_key("llm_stages.hard_translation.enabled", hard_enabled)
                    st.rerun()
                if hard_enabled:
                    _stage_model_input("Difficult-line model", "llm_stages.hard_translation.model")
                    _reasoning_input("Difficult-line reasoning", "llm_stages.hard_translation.reasoning")
                    hard_ratio = st.slider(
                        "Maximum difficult-line percentage",
                        min_value=5,
                        max_value=30,
                        value=int(float(load_key("llm_stages.hard_translation.max_ratio")) * 100),
                        step=5,
                        format="%d%%",
                    )
                    hard_ratio_value = hard_ratio / 100
                    if hard_ratio_value != load_key("llm_stages.hard_translation.max_ratio"):
                        update_key("llm_stages.hard_translation.max_ratio", hard_ratio_value)

                _stage_model_input("Proofreading model", "llm_stages.proofread.model")
                _reasoning_input("Proofreading reasoning", "llm_stages.proofread.reasoning")
                proofread_risky = st.checkbox(
                    "Proofread only risky lines",
                    value=load_key("llm_stages.proofread.only_risky"),
                )
                if proofread_risky != load_key("llm_stages.proofread.only_risky"):
                    update_key("llm_stages.proofread.only_risky", proofread_risky)
                    st.rerun()
    with st.expander(t("Subtitles Settings"), expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            langs = {
                "🇺🇸 English": "en",
                "🇨🇳 简体中文": "zh",
                "🇪🇸 Español": "es",
                "🇷🇺 Русский": "ru",
                "🇫🇷 Français": "fr",
                "🇩🇪 Deutsch": "de",
                "🇮🇹 Italiano": "it",
                "🇯🇵 日本語": "ja",
            }
            lang = st.selectbox(
                t("Recog Lang"),
                options=list(langs.keys()),
                index=list(langs.values()).index(load_key("whisper.language")),
            )
            if langs[lang] != load_key("whisper.language"):
                update_key("whisper.language", langs[lang])
                st.rerun()

        runtime = st.selectbox(
            t("WhisperX Runtime"),
            options=["local", "cloud", "elevenlabs"],
            index=["local", "cloud", "elevenlabs"].index(load_key("whisper.runtime")),
            help=t(
                "Local runtime requires >8GB GPU, cloud runtime requires 302ai API key, elevenlabs runtime requires ElevenLabs API key"
            ),
        )
        if runtime != load_key("whisper.runtime"):
            update_key("whisper.runtime", runtime)
            st.rerun()
        if runtime == "cloud":
            config_input(t("WhisperX 302ai API"), "whisper.whisperX_302_api_key")
        if runtime == "elevenlabs":
            config_input(("ElevenLabs API"), "whisper.elevenlabs_api_key")

        with c2:
            target_language = st.text_input(
                t("Target Lang"),
                value=load_key("target_language"),
                help=t(
                    "Input any language in natural language, as long as llm can understand"
                ),
            )
            if target_language != load_key("target_language"):
                update_key("target_language", target_language)
                st.rerun()

        demucs = st.toggle(
            t("Vocal separation enhance"),
            value=load_key("demucs"),
            help=t(
                "Recommended for videos with loud background noise, but will increase processing time"
            ),
        )
        if demucs != load_key("demucs"):
            update_key("demucs", demucs)
            st.rerun()

        burn_subtitles = st.toggle(
            t("Burn-in Subtitles"),
            value=load_key("burn_subtitles"),
            help=t(
                "Whether to burn subtitles into the video, will increase processing time"
            ),
        )
        if burn_subtitles != load_key("burn_subtitles"):
            update_key("burn_subtitles", burn_subtitles)
            st.rerun()

        reflect_translate = st.toggle(
            t("Reflect Translate"),
            value=load_key("reflect_translate"),
            help=t(
                "Run a second translation pass to make subtitles more natural. Disable for faster local LLM translation."
            ),
        )
        if reflect_translate != load_key("reflect_translate"):
            update_key("reflect_translate", reflect_translate)
            st.rerun()

        pause_after_translate = st.toggle(
            t("Pause After Translate"),
            value=load_key("pause_after_translate"),
            help=t(
                "Pause after generating translation_results.xlsx so you can proofread before the next steps"
            ),
        )
        if pause_after_translate != load_key("pause_after_translate"):
            update_key("pause_after_translate", pause_after_translate)
            st.rerun()

        llm_proofread_enabled = st.toggle(
            t("LLM Proofread Translation"),
            value=load_key("llm_proofread.enabled"),
            help=t(
                "After translation_results.xlsx is generated, ask an LLM to proofread it before manual review."
            ),
        )
        if llm_proofread_enabled != load_key("llm_proofread.enabled"):
            update_key("llm_proofread.enabled", llm_proofread_enabled)
            st.rerun()

        if llm_proofread_enabled:
            proofread_modes = {
                "suggest": t("Suggest only"),
                "apply": t("Apply automatically"),
            }
            proofread_mode = st.selectbox(
                t("LLM Proofread Mode"),
                options=list(proofread_modes.keys()),
                format_func=lambda value: proofread_modes[value],
                index=list(proofread_modes.keys()).index(load_key("llm_proofread.mode"))
                if load_key("llm_proofread.mode") in proofread_modes
                else 0,
                help=t(
                    "Suggest only keeps the original Translation column and adds LLM Proofread. Apply automatically replaces Translation."
                ),
            )
            if proofread_mode != load_key("llm_proofread.mode"):
                update_key("llm_proofread.mode", proofread_mode)
                st.rerun()

            proofread_chunk_lines = st.number_input(
                t("LLM Proofread Chunk Lines"),
                min_value=5,
                max_value=100,
                value=int(load_key("llm_proofread.chunk_lines")),
                step=5,
                help="连续校对行数的目标值；为保留完整句子可超出。每批另带前后文，风险筛选会扩展到完整句子。",
            )
            if proofread_chunk_lines != load_key("llm_proofread.chunk_lines"):
                update_key("llm_proofread.chunk_lines", int(proofread_chunk_lines))

            st.caption("校对附带前后文参考（默认各 4 行，可在 config.yaml 的 llm_proofread.context_lines 调整）；允许同一语义段内跨行调整译文，保留字幕 ID 和时间轴。")
            st.caption("校对表新增 Proofread Status / Issues / Reason，记录模型判定、问题类型及原因。needs_review 表示需人工确认；not_reviewed 表示未送审。已有结果需点击“重新运行 LLM 校对”更新。")
            st.caption("Human Verdict / Human Note 留给人工评价纠错或误改；重跑校对会清空这些评价，请先备份。评价值及多视频统计方法见 docs/translation_quality.md。")

            proofread_override_api = st.toggle(
                t("Override LLM for Proofread"),
                value=load_key("llm_proofread.override_api"),
                help=t(
                    "Use a separate OpenAI-compatible API for proofreading, such as OpenRouter, while keeping the main LLM settings for translation."
                ),
            )
            if proofread_override_api != load_key("llm_proofread.override_api"):
                update_key("llm_proofread.override_api", proofread_override_api)
                st.rerun()

            if proofread_override_api:
                config_input(t("Proofread API_KEY"), "llm_proofread.api.key")
                config_input(
                    t("Proofread BASE_URL"),
                    "llm_proofread.api.base_url",
                    help=t("Openai format, will add /v1/chat/completions automatically"),
                )
                config_input(t("Proofread MODEL"), "llm_proofread.api.model")
                proofread_json = st.toggle(
                    t("Proofread LLM JSON Format Support"),
                    value=load_key("llm_proofread.api.llm_support_json"),
                    help=t("Enable if your proofreading LLM supports JSON mode output"),
                )
                if proofread_json != load_key("llm_proofread.api.llm_support_json"):
                    update_key("llm_proofread.api.llm_support_json", proofread_json)
                    st.rerun()
    with st.expander(t("Dubbing Settings"), expanded=True):
        watermark_enabled = st.toggle(
            t("Enable Text Watermark"),
            value=load_key("watermark.enabled"),
            help=t("Add a text watermark to the final dubbed video"),
        )
        if watermark_enabled != load_key("watermark.enabled"):
            update_key("watermark.enabled", watermark_enabled)
            st.rerun()

        watermark_text = st.text_input(
            t("Watermark Text"),
            value=load_key("watermark.text"),
            help=t("Text shown as the watermark on the final dubbed video"),
        )
        if watermark_text != load_key("watermark.text"):
            update_key("watermark.text", watermark_text)

        position_options = {
            "top_left": t("Top Left"),
            "top_right": t("Top Right"),
            "bottom_left": t("Bottom Left"),
            "bottom_right": t("Bottom Right"),
            "center": t("Center"),
        }
        watermark_position = st.selectbox(
            t("Watermark Position"),
            options=list(position_options.keys()),
            format_func=lambda value: position_options[value],
            index=list(position_options.keys()).index(load_key("watermark.position"))
            if load_key("watermark.position") in position_options
            else 1,
        )
        if watermark_position != load_key("watermark.position"):
            update_key("watermark.position", watermark_position)
            st.rerun()

        watermark_opacity = st.slider(
            t("Watermark Opacity"),
            min_value=0.0,
            max_value=1.0,
            value=float(load_key("watermark.opacity")),
            step=0.05,
        )
        if watermark_opacity != load_key("watermark.opacity"):
            update_key("watermark.opacity", watermark_opacity)

        watermark_font_size = st.slider(
            t("Watermark Size"),
            min_value=12,
            max_value=96,
            value=int(load_key("watermark.font_size")),
            step=2,
        )
        if watermark_font_size != load_key("watermark.font_size"):
            update_key("watermark.font_size", watermark_font_size)

        if watermark_enabled:
            st.caption(t("Watermark Preview"))
            _render_watermark_preview(
                watermark_text,
                watermark_position,
                watermark_opacity,
                watermark_font_size,
            )

        tts_methods = [
            "azure_tts",
            "openai_tts",
            "fish_tts",
            "sf_fish_tts",
            "edge_tts",
            "gpt_sovits",
            "indextts",
            "kokoro_tts",
            "custom_tts",
            "sf_cosyvoice2",
            "f5tts",
        ]
        select_tts = st.selectbox(
            t("TTS Method"),
            options=tts_methods,
            index=tts_methods.index(load_key("tts_method")),
        )
        if select_tts != load_key("tts_method"):
            update_key("tts_method", select_tts)
            st.rerun()

        # sub settings for each tts method
        if select_tts == "sf_fish_tts":
            config_input(t("SiliconFlow API Key"), "sf_fish_tts.api_key")

            # Add mode selection dropdown
            mode_options = {
                "preset": t("Preset"),
                "custom": t("Refer_stable"),
                "dynamic": t("Refer_dynamic"),
            }
            selected_mode = st.selectbox(
                t("Mode Selection"),
                options=list(mode_options.keys()),
                format_func=lambda x: mode_options[x],
                index=list(mode_options.keys()).index(load_key("sf_fish_tts.mode"))
                if load_key("sf_fish_tts.mode") in mode_options.keys()
                else 0,
            )
            if selected_mode != load_key("sf_fish_tts.mode"):
                update_key("sf_fish_tts.mode", selected_mode)
                st.rerun()
            if selected_mode == "preset":
                config_input("Voice", "sf_fish_tts.voice")

        elif select_tts == "openai_tts":
            config_input("302ai API", "openai_tts.api_key")
            config_input(t("OpenAI Voice"), "openai_tts.voice")

        elif select_tts == "fish_tts":
            config_input("302ai API", "fish_tts.api_key")
            fish_tts_character = st.selectbox(
                t("Fish TTS Character"),
                options=list(load_key("fish_tts.character_id_dict").keys()),
                index=list(load_key("fish_tts.character_id_dict").keys()).index(
                    load_key("fish_tts.character")
                ),
            )
            if fish_tts_character != load_key("fish_tts.character"):
                update_key("fish_tts.character", fish_tts_character)
                st.rerun()

        elif select_tts == "azure_tts":
            config_input("302ai API", "azure_tts.api_key")
            config_input(t("Azure Voice"), "azure_tts.voice")

        elif select_tts == "gpt_sovits":
            st.info(t("Please refer to Github homepage for GPT_SoVITS configuration"))
            config_input(t("SoVITS Character"), "gpt_sovits.character")

            refer_mode_options = {
                1: t("Mode 1: Use provided reference audio only"),
                2: t("Mode 2: Use first audio from video as reference"),
                3: t("Mode 3: Use each audio from video as reference"),
            }
            selected_refer_mode = st.selectbox(
                t("Refer Mode"),
                options=list(refer_mode_options.keys()),
                format_func=lambda x: refer_mode_options[x],
                index=list(refer_mode_options.keys()).index(
                    load_key("gpt_sovits.refer_mode")
                ),
                help=t("Configure reference audio mode for GPT-SoVITS"),
            )
            if selected_refer_mode != load_key("gpt_sovits.refer_mode"):
                update_key("gpt_sovits.refer_mode", selected_refer_mode)
                st.rerun()

        elif select_tts == "kokoro_tts":
            kokoro_voices = [
                "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
                "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
            ]
            current_voice = load_key("kokoro_tts.voice")
            selected_voice = st.selectbox(
                "Kokoro Voice",
                options=kokoro_voices,
                index=kokoro_voices.index(current_voice) if current_voice in kokoro_voices else 7,
                help="zf = female, zm = male. These are fixed local voices; no reference audio is used.",
            )
            if selected_voice != current_voice:
                update_key("kokoro_tts.voice", selected_voice)
                st.rerun()

            devices = ["auto", "cuda", "cpu"]
            current_device = str(load_key("kokoro_tts.device"))
            selected_device = st.selectbox(
                "Kokoro Device",
                options=devices,
                index=devices.index(current_device) if current_device in devices else 0,
            )
            if selected_device != current_device:
                update_key("kokoro_tts.device", selected_device)
                st.rerun()

            speed = st.slider(
                "Kokoro Native Speed",
                min_value=0.5,
                max_value=2.0,
                value=float(load_key("kokoro_tts.speed")),
                step=0.05,
                help="1.0 is the natural model speed. VideoLingo still performs final timeline fitting.",
            )
            if speed != load_key("kokoro_tts.speed"):
                update_key("kokoro_tts.speed", speed)

            spell_letters = st.checkbox(
                "Spell Latin acronyms",
                value=bool(load_key("kokoro_tts.spell_latin_letters")),
                help="Read uppercase technical codes letter by letter, e.g. M6 and CTDM.",
            )
            if spell_letters != load_key("kokoro_tts.spell_latin_letters"):
                update_key("kokoro_tts.spell_latin_letters", spell_letters)

        elif select_tts == "indextts":
            version_labels = {"2": "IndexTTS2", "2.5": "IndexTTS2.5"}
            current_version = str(load_key("indextts.version"))
            selected_version = st.selectbox(
                "IndexTTS Version",
                options=list(version_labels.keys()),
                format_func=lambda value: version_labels[value],
                index=list(version_labels.keys()).index(current_version)
                if current_version in version_labels else 0,
                help="The two versions use separate repositories, model folders, and HTTP ports.",
            )
            if selected_version != current_version:
                update_key("indextts.version", selected_version)
                st.rerun()

            profile = "v2_5" if selected_version == "2.5" else "v2"
            st.info(f"Use local {version_labels[selected_version]} with VideoLingo reference audio")
            config_input("IndexTTS Repo Dir", f"indextts.{profile}.repo_dir")
            config_input("IndexTTS Model Dir", f"indextts.{profile}.model_dir")

            use_uv = st.checkbox("Launch with uv", value=load_key("indextts.use_uv"))
            if use_uv != load_key("indextts.use_uv"):
                update_key("indextts.use_uv", use_uv)
                st.rerun()
            if use_uv:
                config_input("uv Path", "indextts.uv_path")
            else:
                config_input("IndexTTS Python", f"indextts.{profile}.python")

            refer_mode_options = {
                2: t("Mode 2: Use first audio from video as reference"),
                3: t("Mode 3: Use each audio from video as reference"),
            }
            selected_refer_mode = st.selectbox(
                t("Refer Mode"),
                options=list(refer_mode_options.keys()),
                format_func=lambda x: refer_mode_options[x],
                index=list(refer_mode_options.keys()).index(
                    load_key("indextts.refer_mode")
                ),
                help="Configure reference audio mode for IndexTTS",
            )
            if selected_refer_mode != load_key("indextts.refer_mode"):
                update_key("indextts.refer_mode", selected_refer_mode)
                st.rerun()

            if selected_version == "2.5":
                use_bf16 = st.checkbox("BF16", value=load_key("indextts.v2_5.bf16"))
                if use_bf16 != load_key("indextts.v2_5.bf16"):
                    update_key("indextts.v2_5.bf16", use_bf16)
                    st.rerun()

                languages = ["ZH", "EN", "JA", "AR", "ES"]
                current_language = str(load_key("indextts.v2_5.language"))
                language = st.selectbox(
                    "IndexTTS Language",
                    options=languages,
                    index=languages.index(current_language) if current_language in languages else 0,
                )
                if language != current_language:
                    update_key("indextts.v2_5.language", language)
                    st.rerun()

                duration_factor = st.slider(
                    "Native duration factor",
                    min_value=0.5,
                    max_value=2.0,
                    value=float(load_key("indextts.v2_5.duration_factor")),
                    step=0.05,
                    help="Below 1.0 speaks faster; above 1.0 speaks slower. Keep 1.0 unless tuning is needed.",
                )
                if duration_factor != load_key("indextts.v2_5.duration_factor"):
                    update_key("indextts.v2_5.duration_factor", duration_factor)

                auto_duration = st.checkbox(
                    "Auto-fit native duration",
                    value=load_key("indextts.v2_5.auto_duration.enabled"),
                    help="Regenerates only lines that exceed their timeline using IndexTTS2.5 native duration control.",
                )
                if auto_duration != load_key("indextts.v2_5.auto_duration.enabled"):
                    update_key("indextts.v2_5.auto_duration.enabled", auto_duration)
                    st.rerun()
                if auto_duration:
                    min_duration_factor = st.slider(
                        "Minimum auto duration factor",
                        min_value=0.5,
                        max_value=1.0,
                        value=float(load_key("indextts.v2_5.auto_duration.min_factor")),
                        step=0.05,
                        help="Lower values allow stronger native compression; 0.75 is the safer default.",
                    )
                    if min_duration_factor != load_key("indextts.v2_5.auto_duration.min_factor"):
                        update_key("indextts.v2_5.auto_duration.min_factor", min_duration_factor)

                use_qwen_emo = st.checkbox(
                    "Load Qwen emotion model",
                    value=load_key("indextts.v2_5.use_qwen_emo"),
                    help="Uses additional VRAM and increases first startup time.",
                )
                if use_qwen_emo != load_key("indextts.v2_5.use_qwen_emo"):
                    update_key("indextts.v2_5.use_qwen_emo", use_qwen_emo)
                    st.rerun()
            else:
                use_fp16 = st.checkbox("FP16", value=load_key("indextts.v2.fp16"))
                if use_fp16 != load_key("indextts.v2.fp16"):
                    update_key("indextts.v2.fp16", use_fp16)
                    st.rerun()

            use_cuda_kernel = st.checkbox("CUDA Kernel", value=load_key("indextts.cuda_kernel"))
            if use_cuda_kernel != load_key("indextts.cuda_kernel"):
                update_key("indextts.cuda_kernel", use_cuda_kernel)
                st.rerun()

            show_console = st.checkbox("Show server console", value=load_key("indextts.show_console"))
            if show_console != load_key("indextts.show_console"):
                update_key("indextts.show_console", show_console)
                st.rerun()

            use_emo_text = st.checkbox("Use emotion from text", value=load_key("indextts.use_emo_text"))
            if use_emo_text != load_key("indextts.use_emo_text"):
                update_key("indextts.use_emo_text", use_emo_text)
                st.rerun()

            emo_alpha = st.slider(
                "Emotion alpha",
                min_value=0.0,
                max_value=1.0,
                value=float(load_key("indextts.emo_alpha")),
                step=0.05,
            )
            if emo_alpha != load_key("indextts.emo_alpha"):
                update_key("indextts.emo_alpha", emo_alpha)
                st.rerun()

        elif select_tts == "edge_tts":
            config_input(t("Edge TTS Voice"), "edge_tts.voice")

        elif select_tts == "sf_cosyvoice2":
            config_input(t("SiliconFlow API Key"), "sf_cosyvoice2.api_key")

        elif select_tts == "f5tts":
            config_input("302ai API", "f5tts.302_api")


def check_api():
    try:
        resp = ask_gpt(
            "This is a test, response 'message':'success' in json format.",
            resp_type="json",
            log_title="None",
        )
        return resp.get("message") == "success"
    except Exception:
        return False


if __name__ == "__main__":
    check_api()
