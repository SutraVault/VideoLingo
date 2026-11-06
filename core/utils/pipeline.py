"""Pipeline operations shared by the interactive workflow and queued workers."""


def operations(host, options):
    return {
        'asr':host['_2_asr'].transcribe,
        'nlp':host['_3_1_split_nlp'].split_by_spacy,
        'meaning':host['_3_2_split_meaning'].split_sentences_by_meaning,
        'terms':host['_4_1_summarize'].get_summary,
        'translation':lambda:(host['_4_2_translate'].translate_all(),host['_4_3_proofread_translation'].proofread_translation_if_enabled()),
        'split':host['_5_split_sub'].split_for_sub_main,
        'subtitles':host['_6_gen_sub'].align_timestamp_main,
        'original':lambda:host['_7_sub_into_vid'].merge_subtitles_to_video(force_burn=True,subtitle_mode=options.original_subtitles),
        'audio_tasks':lambda:(host['_8_1_audio_task'].gen_audio_task_main(),host['_8_2_dub_chunks'].gen_dub_chunks()),
        'references':host['_9_refer_audio'].extract_refer_audio_main,
        'synthesis':host['_10_gen_audio'].gen_audio,
        'merge_audio':host['_11_merge_audio'].merge_full_audio,
        'dubbing':lambda:host['_12_dub_to_vid'].merge_video_audio(subtitle_mode=options.dub_subtitles),
    }
