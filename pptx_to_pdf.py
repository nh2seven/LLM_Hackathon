import subprocess
from pathlib import Path

def convert_pptx_to_pdf(pptx_path: Path):
    output_dir = pptx_path.parent
    cmd = [
        "libreoffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(pptx_path)
    ]
    subprocess.run(cmd, check=True)
    print(f"[✓] Converted {pptx_path.name} to PDF")

if __name__ == "__main__":
    base = Path("LLM_DATASET")
    for pptx in base.glob("**/*.pptx"):
        convert_pptx_to_pdf(pptx)
