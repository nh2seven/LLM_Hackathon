import os
import sys
import argparse
import logging
import glob
import shutil
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import markdown
from markdown_pdf import MarkdownPdf

# Import RAG components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simple_rag.rag_application import OllamaLLM, rag_query, create_topic_vector_store

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime:s) - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
BASE_DATASET_DIR = "LLM_DATASET"
OUTPUT_DIR = "simple_rag/lecture_notes"
MARKDOWN_DIR = "simple_rag/lecture_notes/markdown"
MAX_QUERY_LENGTH = 1000
TOP_K_RESULTS = 6

class LectureNotesGenerator:
    """Generate comprehensive lecture notes with RAG-enhanced content and images"""
    
    def __init__(self, llm, output_dir=OUTPUT_DIR):
        self.llm = llm
        self.output_dir = output_dir
        self.markdown_dir = MARKDOWN_DIR
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.markdown_dir, exist_ok=True)
    
    def _generate_topic_overview(self, topic_name: str, vector_store) -> str:
        """Generate a brief overview of the topic using RAG"""
        query = f"Provide a concise overview of {topic_name} that would be suitable as an introduction in lecture notes. Highlight the key concepts and importance of this topic in computer science or AI."
        
        overview = rag_query(query, vector_store, self.llm, top_k=TOP_K_RESULTS)
        return overview
    
    def _generate_section_content(self, topic_name: str, section_name: str, vector_store) -> str:
        """Generate content for a specific section using RAG"""
        query = f"Provide detailed notes about '{section_name}' in the context of {topic_name}. Include key concepts, examples, and explanations that would be useful for a student studying this topic."
        
        section_content = rag_query(query, vector_store, self.llm, top_k=TOP_K_RESULTS)
        return section_content
    
    def _generate_summary(self, topic_name: str, vector_store) -> str:
        """Generate a summary of the topic using RAG"""
        query = f"Provide a comprehensive summary of {topic_name} that would be suitable as a conclusion in lecture notes. Focus on key takeaways and practical applications."
        
        summary = rag_query(query, vector_store, self.llm, top_k=TOP_K_RESULTS)
        return summary
    
    def _find_topic_images(self, topic_dir: str) -> List[str]:
        """Find all images related to the topic"""
        images_dir = os.path.join(topic_dir, "images")
        if not os.path.exists(images_dir):
            return []
        
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        
        return sorted(image_files)
    
    def _find_captions(self, topic_dir: str, image_name: str, vector_store) -> str:
        """Find captions for images from caption files"""
        caption_files = glob.glob(os.path.join(topic_dir, "*_captions.csv"))
        
        img_basename = os.path.basename(image_name)
        default_caption = f"Figure: {img_basename}"
        
        if not caption_files:
            return default_caption
        
        for caption_file in caption_files:
            try:
                with open(caption_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if img_basename in line:
                            parts = line.strip().split(',', 1)
                            if len(parts) > 1:
                                caption = parts[1].strip('"')
                                if caption and len(caption) > 10:  # Make sure it's not empty or too short
                                    return caption
            except Exception as e:
                logger.error(f"Error reading caption file {caption_file}: {e}")
        
        # If no caption found, try to generate one with RAG
        try:
            img_query = f"Generate a descriptive caption for an image named {img_basename} related to {os.path.basename(topic_dir)}"
            caption = rag_query(img_query, vector_store, self.llm, top_k=3)
            if caption and len(caption) > 10:
                return caption
        except Exception as e:
            logger.error(f"Error generating caption for {img_basename}: {e}")
        
        return default_caption
    
    def _extract_key_sections(self, topic_name: str, vector_store) -> List[str]:
        """Extract key sections for a topic using RAG"""
        query = f"Identify 4-6 key subtopics or sections that should be included in comprehensive lecture notes about {topic_name}. Just list the section names."
        
        sections_text = rag_query(query, vector_store, self.llm, top_k=TOP_K_RESULTS)
        
        # Extract sections from the text
        sections = []
        for line in sections_text.split('\n'):
            line = line.strip()
            # Look for numbered or bulleted lines
            if (line.startswith("- ") or line.startswith("• ") or
                (len(line) > 2 and line[0].isdigit() and line[1] in ['.', ')', ':'])):
                # Clean up the section name
                section = line.split(' ', 1)[1] if ' ' in line else line
                sections.append(section)
            elif ":" in line and not "http" in line:
                # Might be a section with a colon
                section = line.split(":", 1)[0].strip()
                if 3 < len(section) < 50:  # Reasonable length for a section title
                    sections.append(section)
        
        # If we couldn't extract sections properly, use these defaults
        if len(sections) < 3:
            if "tree" in topic_name.lower():
                sections = ["Introduction to Trees", "Tree Traversal Algorithms", 
                            "Implementation Details", "Applications", "Time and Space Complexity"]
            elif "bst" in topic_name.lower() or "binary search" in topic_name.lower():
                sections = ["Binary Search Tree Basics", "BST Operations", 
                            "Implementation Approaches", "Performance Analysis"]
            elif "lora" in topic_name.lower() or "fine" in topic_name.lower():
                sections = ["Introduction to Model Fine-tuning", "LoRA Architecture", 
                            "Quantization Approaches", "Implementation Details", "Applications"]
            elif "mamba" in topic_name.lower() or "multimodal" in topic_name.lower():
                sections = ["Introduction to Model Architecture", "Key Components", 
                            "Training Approaches", "Performance Comparison", "Applications"]
            elif "stable diffusion" in topic_name.lower():
                sections = ["Introduction to Stable Diffusion", "Model Architecture", 
                            "Training Process", "Inference Pipeline", "Applications"]
            elif "agentic" in topic_name.lower():
                sections = ["Introduction to Agentic Systems", "Agentic Workflow Components", 
                            "Implementation Strategies", "Applications", "Future Directions"]
            else:
                sections = ["Introduction", "Core Concepts", "Implementation Details", 
                            "Applications", "Summary"]
        
        return sections[:6]  # Limit to at most 6 sections
    
    def _format_file_path(self, file_path: str) -> str:
        """Format the file path for display in PDF"""
        # Remove absolute path parts but keep the relative structure
        parts = file_path.split(BASE_DATASET_DIR)
        if len(parts) > 1:
            return f"{BASE_DATASET_DIR}{parts[1]}"
        return os.path.basename(file_path)
    
    def _copy_image_to_output(self, image_path: str) -> str:
        """Copy image to output directory for markdown reference"""
        output_images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(output_images_dir, exist_ok=True)
        
        dst_path = os.path.join(output_images_dir, os.path.basename(image_path))
        shutil.copy2(image_path, dst_path)
        
        # Return relative path for markdown
        return os.path.relpath(dst_path, self.markdown_dir)
    
    def _generate_markdown(self, topic_dir: str, vector_store) -> str:
        """Generate markdown content for lecture notes"""
        topic_name = os.path.basename(topic_dir)
        
        # Find images
        images = self._find_topic_images(topic_dir)
        
        # Generate markdown content
        markdown_content = [f"# {topic_name} Lecture Notes\n"]
        
        # Add overview
        overview = self._generate_topic_overview(topic_name, vector_store)
        markdown_content.append("## Overview\n")
        markdown_content.append(f"{overview}\n")
        
        # Add top image if available
        if images:
            img_path = images[0]
            rel_img_path = self._copy_image_to_output(img_path)
            caption = self._find_captions(topic_dir, img_path, vector_store)
            markdown_content.append(f"\n![{caption}]({rel_img_path})\n")
            markdown_content.append(f"*{caption}*\n")
        
        # Generate key sections
        sections = self._extract_key_sections(topic_name, vector_store)
        logger.info(f"Generated {len(sections)} sections for {topic_name}")
        
        img_index = 1  # Start from the second image
        
        # Add section content
        for i, section in enumerate(sections):
            markdown_content.append(f"## {section}\n")
            section_content = self._generate_section_content(topic_name, section, vector_store)
            markdown_content.append(f"{section_content}\n")
            
            # Add an image to the section if available
            if img_index < len(images):
                img_path = images[img_index]
                rel_img_path = self._copy_image_to_output(img_path)
                caption = self._find_captions(topic_dir, img_path, vector_store)
                markdown_content.append(f"\n![{caption}]({rel_img_path})\n")
                markdown_content.append(f"*{caption}*\n")
                img_index += 1
        
        # Add any remaining images
        if img_index < len(images):
            markdown_content.append("## Additional Illustrations\n")
            
            for i in range(img_index, len(images)):
                img_path = images[i]
                rel_img_path = self._copy_image_to_output(img_path)
                caption = self._find_captions(topic_dir, img_path, vector_store)
                markdown_content.append(f"\n![{caption}]({rel_img_path})\n")
                markdown_content.append(f"*{caption}*\n")
        
        # Add summary
        markdown_content.append("## Summary\n")
        summary = self._generate_summary(topic_name, vector_store)
        markdown_content.append(f"{summary}\n")
        
        return "\n".join(markdown_content)
    
    def _convert_markdown_to_pdf(self, markdown_path: str, pdf_path: str) -> bool:
        """Convert markdown file to PDF using markdown_pdf package"""
        try:
            logger.info(f"Converting markdown to PDF using markdown_pdf: {markdown_path} -> {pdf_path}")
            
            # Read the markdown content
            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Create the markdown_pdf converter
            from markdown_pdf import MarkdownPdf, Section
            
            # Initialize the converter
            pdf_converter = MarkdownPdf(toc_level=3)  # Include headers up to level 3 in table of contents
            
            # Create a section with the markdown content
            # Root directory is set to markdown file's directory for proper image resolution
            section = Section(
                text=markdown_content,
                toc=True,
                root=os.path.dirname(markdown_path),
                paper_size='A4'
            )
            
            # Add the section to the PDF
            pdf_converter.add_section(section)
            
            # Save the PDF
            pdf_converter.save(pdf_path)
            
            logger.info(f"Successfully converted markdown to PDF with markdown_pdf: {pdf_path}")
            return True
        except Exception as e:
            logger.error(f"Error during markdown to PDF conversion: {e}")
            return False
    
    def generate_notes(self, topic_dir: str):
        """Generate comprehensive lecture notes for a topic"""
        topic_name = os.path.basename(topic_dir)
        logger.info(f"Generating lecture notes for {topic_name}")
        
        # Create topic-specific vector store
        vector_store = create_topic_vector_store(topic_dir)
        
        # Skip if no documents were found for this topic
        if not vector_store.documents:
            logger.warning(f"No documents found for topic {topic_name}. Skipping note generation.")
            return None
        
        # Determine output paths
        markdown_path = os.path.join(self.markdown_dir, f"{topic_name}_notes.md")
        pdf_path = os.path.join(self.output_dir, f"{topic_name}_notes.pdf")
        
        # Get images for the topic
        images = self._find_topic_images(topic_dir)
        logger.info(f"Found {len(images)} images for {topic_name}")
        
        # Generate markdown content
        markdown_content = self._generate_markdown(topic_dir, vector_store)
        
        # Write markdown to file
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        logger.info(f"Markdown notes saved to {markdown_path}")
        
        # Convert markdown to PDF
        if self._convert_markdown_to_pdf(markdown_path, pdf_path):
            logger.info(f"Lecture notes PDF saved to {pdf_path}")
            return pdf_path
        else:
            logger.warning(f"Failed to convert markdown to PDF for {topic_name}")
            return markdown_path

def main():
    parser = argparse.ArgumentParser(description="Generate lecture notes using RAG")
    parser.add_argument("--topics", nargs="*", help="Specific topics to process (default: all)")
    parser.add_argument("--data_dir", type=str, default=BASE_DATASET_DIR, 
                        help="Base directory containing lecture materials")
    args = parser.parse_args()
    
    # Initialize LLM
    llm = OllamaLLM()
    
    # Initialize notes generator
    notes_generator = LectureNotesGenerator(llm)
    
    # Get topics to process
    if args.topics:
        # Process only specified topics
        topic_dirs = [os.path.join(args.data_dir, topic) for topic in args.topics]
    else:
        # Process all topics in the data directory
        topic_dirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) 
                     if os.path.isdir(os.path.join(args.data_dir, d))]
    
    # Generate notes for each topic
    for topic_dir in topic_dirs:
        if os.path.isdir(topic_dir):
            try:
                notes_generator.generate_notes(topic_dir)
            except Exception as e:
                logger.error(f"Error generating notes for {os.path.basename(topic_dir)}: {e}")
    
    logger.info(f"Completed notes generation for {len(topic_dirs)} topics")

if __name__ == "__main__":
    main()