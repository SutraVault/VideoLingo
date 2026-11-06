from core.utils.project_paths import project_output
# ------------------------------------------
# 定义中间产出文件
# ------------------------------------------

_2_CLEANED_CHUNKS = project_output("output/log/cleaned_chunks.xlsx")
_SOURCE_SUBTITLE_SEGMENTS = project_output("output/log/source_subtitle_segments.xlsx")
_3_1_SPLIT_BY_NLP = project_output("output/log/split_by_nlp.txt")
_3_2_SPLIT_BY_MEANING = project_output("output/log/split_by_meaning.txt")
_4_1_TERMINOLOGY = project_output("output/log/terminology.json")
_4_2_TRANSLATION = project_output("output/log/translation_results.xlsx")
_4_3_PROOFREAD_TRANSLATION = project_output("output/log/translation_results_proofread.xlsx")
_5_SPLIT_SUB = project_output("output/log/translation_results_for_subtitles.xlsx")
_5_REMERGED = project_output("output/log/translation_results_remerged.xlsx")

_8_1_AUDIO_TASK = project_output("output/audio/tts_tasks.xlsx")


# ------------------------------------------
# 定义音频文件
# ------------------------------------------
_OUTPUT_DIR = project_output("output")
_AUDIO_DIR = project_output("output/audio")
_RAW_AUDIO_FILE = project_output("output/audio/raw.mp3")
_VOCAL_AUDIO_FILE = project_output("output/audio/vocal.mp3")
_BACKGROUND_AUDIO_FILE = project_output("output/audio/background.mp3")
_AUDIO_REFERS_DIR = project_output("output/audio/refers")
_AUDIO_SEGS_DIR = project_output("output/audio/segs")
_AUDIO_TMP_DIR = project_output("output/audio/tmp")

# ------------------------------------------
# 导出
# ------------------------------------------

__all__ = [
    "_2_CLEANED_CHUNKS",
    "_SOURCE_SUBTITLE_SEGMENTS",
    "_3_1_SPLIT_BY_NLP",
    "_3_2_SPLIT_BY_MEANING",
    "_4_1_TERMINOLOGY",
    "_4_2_TRANSLATION",
    "_4_3_PROOFREAD_TRANSLATION",
    "_5_SPLIT_SUB",
    "_5_REMERGED",
    "_8_1_AUDIO_TASK",
    "_OUTPUT_DIR",
    "_AUDIO_DIR",
    "_RAW_AUDIO_FILE",
    "_VOCAL_AUDIO_FILE",
    "_BACKGROUND_AUDIO_FILE",
    "_AUDIO_REFERS_DIR",
    "_AUDIO_SEGS_DIR",
    "_AUDIO_TMP_DIR"
]
