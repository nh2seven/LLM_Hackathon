# Lecture Notes Generator
This project creates comprehensive lecture notes from various source materials using RAG (Retrieval-Augmented Generation) with Ollama.

## Overview
The pipeline consists of five main stages:

1. **Converting PPTX to PDF**: Converts PowerPoint presentations to PDF format.
2. **Audio Extraction**: Extracts audio from video lectures.
3. **Audio Transcription**: Transcribes audio to text using OpenAI's Whisper model.
4. **PDF Processing**: Extracts text and images from PDFs with automatic caption generation.
5. **RAG Notes Generation**: Creates comprehensive, well-structured lecture notes using RAG with Ollama.

## Requirements
- Python 3.8+ with required packages (listed in requirements.txt)
- LibreOffice (for PPTX to PDF conversion)
- FFmpeg (for audio extraction)
- Ollama with the `mistral:latest` model
> NOTE: Works best on a pure Linux system.

## Installation
1. Install the required Python packages:
```bash
pip install -r requirements.txt
```

2. Install external dependencies:
```bash
# On Debian/Ubuntu
sudo apt-get update
sudo apt-get install ffmpeg libreoffice

# On macOS
brew install ffmpeg libreoffice
```

3. Install Ollama from [https://ollama.ai/](https://ollama.ai/) and pull the Mistral model:
```bash
ollama pull mistral:latest
```

## Usage
### End-to-End Pipeline
Run the entire pipeline with:
```bash
python scripts/generate_all_notes.py
```
This will:
- Convert all PPTX files to PDF
- Extract audio from all video files
- Transcribe all audio files
- Process all PDF files
- Generate lecture notes for all topics

### Process Specific Topics
To process only specific topics:
```bash
python scripts/generate_all_notes.py --topics BST binary_tree_traversal
```

### Skip Specific Steps
You can skip specific pipeline steps:
```bash
# Skip PPTX conversion
python scripts/generate_all_notes.py --skip-pptx

# Skip audio extraction and transcription
python scripts/generate_all_notes.py --skip-audio

# Skip PDF processing
python scripts/generate_all_notes.py --skip-pdf

# Only run the notes generator
python scripts/generate_all_notes.py --only-notes
```

## Pipeline Components
The project consists of the following main components:
- `pptx_to_pdf.py`: Converts PPTX files to PDF
- `audio.py`: Extracts audio from video files
- `transcribe.py`: Transcribes audio files to text
- `extraction.py`: Extracts content from PDF files
- `simple_rag/`: Contains RAG-related components
  - `vector_store.py`: TF-IDF based vector store
  - `rag_application.py`: RAG query system using Ollama
  - `lecture_notes_generator.py`: Generates lecture notes

## Output
The generated notes will be saved in:
- Markdown: `simple_rag/lecture_notes/markdown/`
- PDF: `simple_rag/lecture_notes/`

## Data Structure
Place your lecture materials in `LLM_DATASET/` with one folder per topic:
```
LLM_DATASET/
└── topic_name/
    ├── presentation.pptx
    ├── presentation.pdf
    ├── lecture.mp4
    ├── mp3/
    │   └── lecture.mp3
    ├── lecture.mp3.txt
    ├── presentation_gen.txt
    ├── presentation_captions.csv
    └── images/
        └── img_1_1.png
```

## Troubleshooting
- **Ollama Connection**: Ensure the Ollama service is running on http://localhost:11434
- **PDF Extraction**: If PDF extraction fails, check that the PDF is not password-protected
- **LibreOffice**: If PPTX conversion fails, ensure LibreOffice is correctly installed