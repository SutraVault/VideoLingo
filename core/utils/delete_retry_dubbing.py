import os
import shutil


def _remove_path(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        return True
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False


def rebuild_audio_subtitles_and_clear_dubbing():
    """Rebuild the reviewed audio SRT, then clear only derived dubbing artifacts."""
    # Import lazily to avoid a circular import while the core package initializes.
    from core._6_gen_sub import align_timestamp_main

    # Do this first: if subtitle rebuilding fails, existing audio remains intact.
    align_timestamp_main()

    targets = [
        os.path.join("output", "audio", "tts_tasks.xlsx"),
        os.path.join("output", "audio", "tmp"),
        os.path.join("output", "audio", "segs"),
        os.path.join("output", "dub.wav"),
        os.path.join("output", "output_dub.mp4"),
    ]
    removed = []
    for path in targets:
        if _remove_path(path):
            removed.append(path)
    return removed

def delete_dubbing_files():
    files_to_delete = [
        os.path.join("output", "dub.wav"),
        os.path.join("output", "output_dub.mp4")
    ]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
        else:
            print(f"File not found: {file_path}")
    
    segs_folder = os.path.join("output", "audio", "segs")
    if os.path.exists(segs_folder):
        try:
            shutil.rmtree(segs_folder)
            print(f"Deleted folder and contents: {segs_folder}")
        except Exception as e:
            print(f"Error deleting folder {segs_folder}: {str(e)}")
    else:
        print(f"Folder not found: {segs_folder}")

if __name__ == "__main__":
    delete_dubbing_files()
