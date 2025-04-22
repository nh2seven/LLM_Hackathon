#!/usr/bin/env python3
"""
End-to-end script for generating lecture notes from various sources.

This script orchestrates the entire pipeline:
1. Convert PPTX files to PDF
2. Extract audio from videos
3. Transcribe audio to text
4. Extract content from PDFs
5. Generate RAG-enhanced lecture notes

Usage:
    python generate_all_notes.py [--topics TOPIC1 TOPIC2 ...]
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all pipeline components
from pptx_to_pdf import convert_pptx_to_pdf
from audio import extract as extract_audio
import transcribe  # This runs automatically when imported
from extraction import batch_extract_pdfs
from simple_rag.lecture_notes_generator import main as generate_notes

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("lecture_notes_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATASET_DIR = "LLM_DATASET"
OLLAMA_MODEL = "mistral:latest"

def check_dependencies():
    """Check if required external tools are installed"""
    dependencies = {
        "ffmpeg": ["ffmpeg", "-version"],
        "libreoffice": ["libreoffice", "--version"],
        "ollama": ["ollama", "list"]
    }
    
    missing = []
    
    for name, cmd in dependencies.items():
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            logger.info(f"✓ {name} is installed")
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error(f"✗ {name} is not installed or not working correctly")
            missing.append(name)
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        return False
    
    return True

def check_ollama_model():
    """Check if the required Ollama model is available"""
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True,
            text=True
        )
        
        if OLLAMA_MODEL in result.stdout:
            logger.info(f"✓ Ollama model {OLLAMA_MODEL} is available")
            return True
        else:
            logger.warning(f"✗ Ollama model {OLLAMA_MODEL} is not available")
            
            # Try to pull the model
            logger.info(f"Pulling Ollama model {OLLAMA_MODEL}...")
            subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
            return True
    except Exception as e:
        logger.error(f"Error checking Ollama model: {e}")
        return False

def convert_all_pptx():
    """Convert all PPTX files to PDF"""
    logger.info("Step 1: Converting PPTX files to PDF")
    base = Path(DATASET_DIR)
    count = 0
    
    for pptx in base.glob("**/*.pptx"):
        try:
            convert_pptx_to_pdf(pptx)
            count += 1
        except Exception as e:
            logger.error(f"Error converting {pptx}: {e}")
    
    logger.info(f"Converted {count} PPTX files to PDF")

def process_audio_files():
    """Extract audio from video files and transcribe them"""
    logger.info("Step 2: Extracting audio from video files")
    extract_audio()
    
    logger.info("Step 3: Transcribing audio files")
    # transcribe.py runs its logic when imported, no need to call anything

def process_pdf_files():
    """Extract content from PDF files"""
    logger.info("Step 4: Extracting content from PDF files")
    batch_extract_pdfs(DATASET_DIR)

def run_lecture_notes_generator(topics=None):
    """Generate lecture notes using RAG"""
    logger.info("Step 5: Generating RAG-enhanced lecture notes")
    
    # Create a mock args object to pass to the main function
    class Args:
        pass
    
    args = Args()
    args.data_dir = DATASET_DIR
    args.topics = topics
    
    # Monkey patch sys.argv to pass args to the main function
    orig_argv = sys.argv
    sys.argv = [sys.argv[0]]
    
    if topics:
        sys.argv.extend(["--topics"] + topics)
    
    try:
        generate_notes()
    finally:
        # Restore original sys.argv
        sys.argv = orig_argv

def main():
    parser = argparse.ArgumentParser(description="End-to-end lecture notes generation pipeline")
    parser.add_argument("--topics", nargs="*", help="Specific topics to process (default: all)")
    parser.add_argument("--skip-pptx", action="store_true", help="Skip PPTX to PDF conversion")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio extraction and transcription")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF content extraction")
    parser.add_argument("--only-notes", action="store_true", help="Only run the notes generator")
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing required dependencies. Please install them and try again.")
        return 1
    
    # Check if Ollama model is available
    if not check_ollama_model():
        logger.error(f"Ollama model {OLLAMA_MODEL} is required but not available.")
        return 1
    
    # Create dataset directory if it doesn't exist
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    if args.only_notes:
        run_lecture_notes_generator(args.topics)
        return 0
    
    # Run the pipeline steps
    if not args.skip_pptx:
        convert_all_pptx()
    
    if not args.skip_audio:
        process_audio_files()
    
    if not args.skip_pdf:
        process_pdf_files()
    
    # Generate lecture notes
    run_lecture_notes_generator(args.topics)
    
    logger.info("✓ Lecture notes generation pipeline completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())