from pathlib import Path
import whisper


# Use any OpenAI Whisper model
model = whisper.load_model("turbo")
root = Path("LLM_DATASET")
mp3_files = list(root.glob("**/mp3/*.mp3"))

for mp3_file in mp3_files:
    # Save the transcript as a .txt file in the same dir as the .mp3 file for simplicity
    txt_file = mp3_file.with_suffix(".txt")
    
    # Check if the text file already exists
    if txt_file.exists():
        print(f"[*] Skipping: {mp3_file.relative_to(root)} (transcript already exists)")
        continue
    
    print(f"[+] Transcribing: {mp3_file.relative_to(root)}")
    
    # Transcribe the audio
    result = model.transcribe(str(mp3_file))
    txt_file.write_text(result["text"], encoding="utf-8")

    print(f"    └── Saved transcript to: {txt_file.relative_to(root)}")
