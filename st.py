import streamlit as st
import hashlib
import json
import os, sys, time
import streamlit.components.v1 as components
from core.st_utils.imports_and_utils import *
from core.st_utils.task_runner import TaskRunner
from core.utils.srt_import import import_srt_to_cleaned_chunks
from core import *

# SET PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + current_dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="VideoLingo", page_icon="docs/logo.svg")

SUB_VIDEO = "output/output_sub.mp4"
DUB_VIDEO = "output/output_dub.mp4"
TRANSLATION_RESULTS = "output/log/translation_results.xlsx"
TRANS_SRT = "output/trans.srt"
SOURCE_SUBTITLE = "output/source_subtitle.srt"
CLEANED_CHUNKS = "output/log/cleaned_chunks.xlsx"


def _notify_user_handoff(notification_key: str, title: str, message: str, kind: str = "info"):
    """Play a short browser tone and request/show a desktop notification once."""
    session_key = f"_handoff_notification_{notification_key}"
    if st.session_state.get(session_key):
        return

    st.session_state[session_key] = True
    if sys.platform.startswith("win"):
        try:
            import winsound

            if kind == "error":
                winsound.MessageBeep(winsound.MB_ICONHAND)
            else:
                winsound.MessageBeep(winsound.MB_ICONASTERISK)
        except RuntimeError:
            pass
    st.toast(message, icon="❌" if kind == "error" else "🔔")
    components.html(
        f"""
        <script>
        const title = {json.dumps(title)};
        const message = {json.dumps(message)};
        const tag = {json.dumps(notification_key)};
        const kind = {json.dumps(kind)};

        function playTone(context, frequency, start, duration) {{
            const oscillator = context.createOscillator();
            const gain = context.createGain();
            oscillator.type = kind === "error" ? "square" : "sine";
            oscillator.frequency.setValueAtTime(frequency, context.currentTime + start);
            gain.gain.setValueAtTime(0.001, context.currentTime + start);
            gain.gain.exponentialRampToValueAtTime(0.25, context.currentTime + start + 0.02);
            gain.gain.exponentialRampToValueAtTime(0.001, context.currentTime + start + duration);
            oscillator.connect(gain);
            gain.connect(context.destination);
            oscillator.start(context.currentTime + start);
            oscillator.stop(context.currentTime + start + duration);
        }}

        function ringBell() {{
            try {{
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                if (!AudioContext) return;
                const context = new AudioContext();
                if (kind === "error") {{
                    playTone(context, 330, 0, 0.25);
                    playTone(context, 220, 0.28, 0.35);
                }} else {{
                    playTone(context, 880, 0, 0.5);
                }}
            }} catch (error) {{}}
        }}

        function showNotification() {{
            if (!("Notification" in window)) return;
            const options = {{ body: message, tag, renotify: true }};
            if (Notification.permission === "granted") {{
                new Notification(title, options);
            }} else if (Notification.permission !== "denied") {{
                Notification.requestPermission().then((permission) => {{
                    if (permission === "granted") new Notification(title, options);
                }});
            }}
        }}

        ringBell();
        showNotification();
        </script>
        """,
        height=0,
    )


def _decode_uploaded_text(uploaded_file) -> str:
    raw = uploaded_file.getvalue()
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode subtitle file.")


def _remove_text_outputs(remove_cleaned_chunks=False):
    files = [
        "output/log/split_by_nlp.txt",
        "output/log/split_by_meaning.txt",
        "output/log/terminology.json",
        "output/log/translation_results.xlsx",
        "output/log/translation_results_for_subtitles.xlsx",
        "output/log/translation_results_remerged.xlsx",
        "output/src.srt",
        "output/trans.srt",
        "output/src_trans.srt",
        "output/trans_src.srt",
        "output/output_sub.mp4",
        "output/audio/src_subs_for_audio.srt",
        "output/audio/trans_subs_for_audio.srt",
        "output/audio/tts_tasks.xlsx",
        "output/dub.srt",
        "output/dub.wav",
        "output/output_dub.mp4",
    ]
    if remove_cleaned_chunks:
        files.extend([CLEANED_CHUNKS, SOURCE_SUBTITLE])

    for path in files:
        if os.path.exists(path):
            os.remove(path)


# ─── Task control UI (auto-refreshes every 1s while task is active) ───


@st.fragment(run_every=1)
def _task_control_panel(runner_key: str):
    """Renders progress bar + pause/stop buttons. Auto-refreshes every 1s."""
    runner = TaskRunner.get(st.session_state, runner_key)

    if runner.state == "idle":
        return

    # Progress
    step_text = (
        f"({runner.current_step + 1}/{runner.total_steps}) {runner.current_label}"
        if runner.current_step >= 0
        else ""
    )

    if runner.is_active:
        if runner.state == "paused":
            st.warning(f"⏸️ {t('Paused')} {step_text}")
        else:
            st.info(f"⏳ {t('Running...')} {step_text}")
        st.progress(runner.progress)

        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if runner.state == "paused":
                if st.button(
                    f"▶️ {t('Resume')}",
                    key=f"{runner_key}_resume",
                    use_container_width=True,
                ):
                    runner.resume()
                    st.rerun()
            else:
                if st.button(
                    f"⏸️ {t('Pause')}",
                    key=f"{runner_key}_pause",
                    use_container_width=True,
                ):
                    runner.pause()
                    st.rerun()
        with col2:
            if st.button(
                f"⏹️ {t('Stop')}",
                key=f"{runner_key}_stop",
                use_container_width=True,
                type="primary",
            ):
                runner.stop()
                st.rerun()

    elif runner.state == "completed":
        st.success(t("Task completed!"))
        st.progress(1.0)
        runner.reset()
        time.sleep(0.5)
        st.rerun(scope="app")

    elif runner.state == "stopped":
        st.warning(f"⏹️ {t('Task stopped')} {step_text}")
        if st.button(t("OK"), key=f"{runner_key}_ack_stop", use_container_width=True):
            runner.reset()
            st.rerun(scope="app")

    elif runner.state == "error":
        error_key = hashlib.sha1(
            f"{runner_key}|{runner.current_step}|{runner.current_label}|{runner.error_msg}".encode(
                "utf-8", errors="ignore"
            )
        ).hexdigest()[:12]
        _notify_user_handoff(
            f"task_error_{error_key}",
            "VideoLingo",
            f"{t('Task error')}: {runner.current_label or step_text}",
            kind="error",
        )
        st.error(f"❌ {t('Task error')}: {runner.error_msg}")
        if st.button(t("OK"), key=f"{runner_key}_ack_error", use_container_width=True):
            runner.reset()
            st.rerun(scope="app")


# ─── Text processing ───


def _get_text_steps(merge_subtitles=False, until_translation=False, after_translation=False):
    """Return the subtitle processing steps as (label, callable) list."""
    steps = [
        (t("WhisperX word-level transcription"), _2_asr.transcribe),
        (
            t("Sentence segmentation using NLP and LLM"),
            lambda: (
                _3_1_split_nlp.split_by_spacy(),
                _3_2_split_meaning.split_sentences_by_meaning(),
            ),
        ),
        (
            t("Summarization and multi-step translation"),
            lambda: (_4_1_summarize.get_summary(), _4_2_translate.translate_all()),
        ),
    ]

    if until_translation:
        return steps

    remaining_steps = [
        (t("Cutting and aligning long subtitles"), _5_split_sub.split_for_sub_main),
        (t("Generating timeline and subtitles"), _6_gen_sub.align_timestamp_main),
    ]

    if merge_subtitles:
        remaining_steps.append(
            (
                t("Merging subtitles into the video"),
                lambda: _7_sub_into_vid.merge_subtitles_to_video(force_burn=True),
            )
        )
    elif os.path.exists(SUB_VIDEO):
        remaining_steps.append(
            (
                t("Merging subtitles into the video"),
                lambda: os.remove(SUB_VIDEO),
            )
        )

    return remaining_steps if after_translation else steps + remaining_steps


def _text_step_labels(merge_subtitles=False):
    labels = [
        t("WhisperX word-level transcription"),
        t("Sentence segmentation using NLP and LLM"),
        t("Summarization and multi-step translation"),
        t("Cutting and aligning long subtitles"),
        t("Generating timeline and subtitles"),
    ]
    if merge_subtitles:
        labels.append(t("Merging subtitles into the video"))
    return labels


def _text_step_status(index, label, runner, translation_ready, subtitles_ready):
    if subtitles_ready:
        return "complete"
    if index == 0 and os.path.exists(CLEANED_CHUNKS):
        return "skipped"
    if runner.is_active and runner.current_label == label:
        return runner.state
    if runner.is_active and runner.current_step >= 0:
        labels = [step_label for step_label, _ in runner._steps]
        if label in labels and labels.index(label) < runner.current_step:
            return "complete"
    if translation_ready and index < 3:
        return "complete"
    return "pending"


def _render_step_row(index, label, status):
    status_mark = {
        "complete": "[done]",
        "running": "[running]",
        "paused": "[paused]",
        "skipped": "[skipped]",
        "pending": "[pending]",
    }.get(status, "[pending]")
    if status == "running":
        st.info(f"{index + 1}. {label} {status_mark}")
    elif status == "paused":
        st.warning(f"{index + 1}. {label} {status_mark}")
    elif status == "complete":
        st.success(f"{index + 1}. {label} {status_mark}")
    elif status == "skipped":
        st.info(f"{index + 1}. {label} {status_mark}")
    else:
        st.write(f"{index + 1}. {label} {status_mark}")


def _render_text_steps(runner, merge_subtitles=False):
    labels = _text_step_labels(merge_subtitles)
    translation_ready = os.path.exists(TRANSLATION_RESULTS)
    subtitles_ready = os.path.exists(TRANS_SRT)

    for index, label in enumerate(labels):
        if index == 0 and load_key("pause_after_translate") and not translation_ready and not runner.is_active:
            if st.button(t("Generate Translation Draft"), key="text_processing_button"):
                runner.start(_get_text_steps(until_translation=True))
                st.rerun()

        if index == 3 and load_key("pause_after_translate") and translation_ready and not subtitles_ready and not runner.is_active:
            _notify_user_handoff(
                f"translation_draft_{os.path.getmtime(TRANSLATION_RESULTS)}",
                "VideoLingo",
                t(
                    "Translation draft ready. Proofread `output/log/translation_results.xlsx`, then click Continue After Review."
                ),
            )
            st.info(
                t(
                    "translation_results.xlsx is ready. Proofread it, then continue the remaining subtitle steps here."
                )
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    t("Continue After Review"),
                    key="continue_text_processing_button",
                ):
                    steps = _get_text_steps(
                        merge_subtitles=merge_subtitles,
                        after_translation=True,
                    )
                    runner.start(steps)
                    st.rerun()
            with col2:
                if st.button(
                    t("Delete Translation Draft"),
                    key="delete_translation_draft_button",
                ):
                    os.remove(TRANSLATION_RESULTS)
                    st.rerun()

        status = _text_step_status(index, label, runner, translation_ready, subtitles_ready)
        _render_step_row(index, label, status)

    if runner.is_active:
        st.progress(runner.progress)


def text_processing_section():
    st.header(t("b. Translate and Generate Subtitles"))
    runner = TaskRunner.get(st.session_state, "_text_runner")

    with st.container(border=True):
        st.markdown(f"<p style='font-size: 20px;'>{t('This stage includes the following steps:')}</p>", unsafe_allow_html=True)

        merge_subtitles = st.checkbox(
            t("Merging subtitles into the video"),
            value=False,
            key="text_processing_merge_subtitles",
        )
        auto_start_dubbing = st.checkbox(
            t("Auto-start dubbing after subtitles complete"),
            value=load_key("auto_start_dubbing_after_subtitles"),
            key="text_processing_auto_start_dubbing",
            help=t(
                "When subtitle processing finishes, automatically start stage c. Dubbing."
            ),
        )
        if auto_start_dubbing != load_key("auto_start_dubbing_after_subtitles"):
            update_key("auto_start_dubbing_after_subtitles", auto_start_dubbing)
            st.rerun()

        if os.path.exists(CLEANED_CHUNKS) and not os.path.exists(TRANS_SRT):
            st.info(
                t(
                    "Existing transcription data found. WhisperX word-level transcription will be skipped."
                )
            )
        _render_text_steps(runner, merge_subtitles)

        if not os.path.exists(TRANS_SRT):
            if runner.is_active:
                _task_control_panel("_text_runner")
            elif runner.is_done:
                _task_control_panel("_text_runner")
            elif load_key("pause_after_translate"):
                pass
            else:
                if st.button(
                    t("Start Processing Subtitles"), key="text_processing_button"
                ):
                    steps = _get_text_steps(merge_subtitles=merge_subtitles)
                    runner.start(steps)
                    st.rerun()
        else:
            if os.path.exists(SUB_VIDEO):
                st.video(SUB_VIDEO)
            download_subtitle_zip_button(text=t("Download All Srt Files"))

            if st.button(t("Archive to 'history'"), key="cleanup_in_text_processing"):
                cleanup()
                st.rerun()
            return True


def source_subtitle_section():
    st.header(t("Optional: Use Existing Source Subtitles"))

    with st.container(border=True):
        st.write(
            t(
                "Upload a source-language SRT file to skip WhisperX transcription and translate from the existing subtitle timeline."
            )
        )

        imported_subtitle_ready = os.path.exists(SOURCE_SUBTITLE) and os.path.exists(CLEANED_CHUNKS)
        cleaned_chunks_ready = os.path.exists(CLEANED_CHUNKS)

        if imported_subtitle_ready:
            st.success(
                t(
                    "Source subtitles imported. WhisperX will be skipped when subtitle processing starts."
                )
            )
            if st.button(t("Remove Imported Source Subtitles"), key="remove_source_subtitle_button"):
                _remove_text_outputs(remove_cleaned_chunks=True)
                st.rerun()
            return

        if cleaned_chunks_ready:
            st.info(
                t(
                    "Transcription data already exists. Subtitle processing will reuse it and skip WhisperX."
                )
            )

        uploaded_subtitle = st.file_uploader(
            t("Upload source SRT subtitle"),
            type=["srt"],
            key="source_subtitle_uploader",
        )
        if uploaded_subtitle and st.button(
            t("Import Source Subtitles"),
            key="import_source_subtitle_button",
        ):
            try:
                content = _decode_uploaded_text(uploaded_subtitle)
                _remove_text_outputs(remove_cleaned_chunks=False)
                os.makedirs("output", exist_ok=True)
                with open(SOURCE_SUBTITLE, "w", encoding="utf-8") as file:
                    file.write(content)
                _, subtitle_count, word_count = import_srt_to_cleaned_chunks(content)
                st.success(
                    t("Imported {subtitle_count} subtitle blocks and {word_count} word timestamps.").format(
                        subtitle_count=subtitle_count,
                        word_count=word_count,
                    )
                )
                time.sleep(0.5)
                st.rerun()
            except Exception as exc:
                st.error(f"{t('Failed to import source subtitles')}: {exc}")


# ─── Audio processing ───


def _get_audio_steps():
    """Return the audio/dubbing processing steps as (label, callable) list."""
    steps = [
        (
            t("Generate audio tasks and chunks"),
            lambda: (
                _8_1_audio_task.gen_audio_task_main(),
                _8_2_dub_chunks.gen_dub_chunks(),
            ),
        ),
        (t("Extract reference audio"), _9_refer_audio.extract_refer_audio_main),
        (t("Generate and merge audio files"), _10_gen_audio.gen_audio),
        (t("Merge full audio"), _11_merge_audio.merge_full_audio),
        (t("Merge final audio into video"), _12_dub_to_vid.merge_video_audio),
    ]
    return steps


def audio_processing_section():
    st.header(t("c. Dubbing"))
    runner = TaskRunner.get(st.session_state, "_audio_runner")

    with st.container(border=True):
        st.markdown(
            f"""
        <p style='font-size: 20px;'>
        {t("This stage includes the following steps:")}
        <p style='font-size: 20px;'>
            1. {t("Generate audio tasks and chunks")}<br>
            2. {t("Extract reference audio")}<br>
            3. {t("Generate and merge audio files")}<br>
            4. {t("Merge final audio into video")}
        """,
            unsafe_allow_html=True,
        )

        if not os.path.exists(DUB_VIDEO):
            if os.path.exists(TRANS_SRT) and not runner.is_active and not runner.is_done:
                _notify_user_handoff(
                    f"audio_ready_{os.path.getmtime(TRANS_SRT)}",
                    "VideoLingo",
                    t("Subtitle processing complete! 🎉"),
                )
            if runner.is_active:
                _task_control_panel("_audio_runner")
            elif runner.is_done:
                _task_control_panel("_audio_runner")
            elif os.path.exists(TRANS_SRT) and load_key(
                "auto_start_dubbing_after_subtitles"
            ):
                st.info(
                    t(
                        "Auto-starting dubbing because subtitle processing is complete."
                    )
                )
                runner.start(_get_audio_steps())
                st.rerun()
            else:
                if st.button(
                    t("Start Audio Processing"), key="audio_processing_button"
                ):
                    steps = _get_audio_steps()
                    runner.start(steps)
                    st.rerun()
        else:
            st.success(
                t(
                    "Audio processing is complete! You can check the audio files in the `output` folder."
                )
            )
            if load_key("burn_subtitles"):
                st.video(DUB_VIDEO)
            if st.button(t("Delete dubbing files"), key="delete_dubbing_files"):
                delete_dubbing_files()
                st.rerun()
            if st.button(t("Archive to 'history'"), key="cleanup_in_audio_processing"):
                cleanup()
                st.rerun()


# ─── Main ───


def main():
    logo_col, _ = st.columns([1, 1])
    with logo_col:
        st.image("docs/logo.png", width="stretch")
    st.markdown(button_style, unsafe_allow_html=True)
    welcome_text = t(
        'Hello, welcome to VideoLingo. If you encounter any issues, feel free to get instant answers with our Free QA Agent <a href="https://share.fastgpt.in/chat/share?shareId=066w11n3r9aq6879r4z0v9rh" target="_blank">here</a>! You can also try out our SaaS website at <a href="https://videolingo.io" target="_blank">videolingo.io</a> for free!'
    )
    st.markdown(
        f"<p style='font-size: 20px; color: #808080;'>{welcome_text}</p>",
        unsafe_allow_html=True,
    )
    # add settings
    with st.sidebar:
        page_setting()
        st.markdown(give_star_button, unsafe_allow_html=True)
    video_ready = download_video_section()
    if video_ready:
        source_subtitle_section()
        text_processing_section()
        audio_processing_section()


if __name__ == "__main__":
    main()
