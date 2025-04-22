import subprocess
from pathlib import Path

VIDEO_EXTS = [".mp4", ".mkv"]
ROOT_DIR = "LLM_DATASET"


# This script extracts audio from video files in the specified directory
def audio(video_path: Path, output_path: Path):
    output_path.parent.mkdir(exist_ok=True, parents=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",  # no video
        "-codec:a",
        "libmp3lame",
        "-qscale:a",
        "2",
        str(output_path),
    ]
    print(f"[+] Extracting MP3 from {video_path.name}")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# This function iterates through all subdirectories in the ROOT_DIR
def extract():
    root = Path(ROOT_DIR)
    for subdir in root.iterdir():
        if subdir.is_dir():
            for file in subdir.iterdir():
                if file.suffix.lower() in VIDEO_EXTS:
                    out_file = subdir / "mp3" / (file.stem + ".mp3")
                    if not out_file.exists():
                        audio(file, out_file)
                    else:
                        print(f"[!] Skipping (already exists): {out_file.name}")


if __name__ == "__main__":
    extract()
