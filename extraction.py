import hashlib
import fitz  # PyMuPDF
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import cm
from transformers import pipeline
import torch
from pathlib import Path
import csv

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Lazy loading for DistilBART model
_summarizer = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)
    return _summarizer


def hash_image_bytes(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()


def extract_pdf_content_filtered(pdf_path):
    doc = fitz.open(pdf_path)
    all_slides = []
    seen_hashes = set()
    
    parent_dir = Path(pdf_path).parent
    image_dir = parent_dir / "images"
    image_dir.mkdir(exist_ok=True)

    for page_num, page in enumerate(doc):
        slide = {'slide_number': page_num + 1, 'title': None, 'content': '', 'images': []}
        blocks = page.get_text("dict")["blocks"]
        texts = []

        for b in blocks:
            if "lines" in b:
                for l in b["lines"]:
                    line_text = " ".join([span["text"] for span in l["spans"]])
                    texts.append(line_text)

        if texts:
            slide['title'] = texts[0]
            slide['content'] = "\n".join(texts[1:])

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_hash = hash_image_bytes(image_bytes)

            if image_hash in seen_hashes:
                continue

            seen_hashes.add(image_hash)
            image_ext = base_image["ext"]
            img_filename = f"img_{page_num + 1}_{img_index}.{image_ext}"
            image_path = image_dir / img_filename

            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            slide['images'].append(str(image_path))

        all_slides.append(slide)

    return all_slides


def generate_caption_with_distilbart(slide_text):
    """Generate a caption using DistilBART summarization model"""
    if not slide_text.strip():
        return "No description available"
        
    summarizer = get_summarizer()
    try:
        summary = summarizer(slide_text, max_length=25, min_length=10, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "Caption generation failed"


def get_caption_from_descriptions(image_path, image_descriptions):
    """Get caption for an image from the image descriptions list"""
    for path, caption in image_descriptions:
        if path == image_path:
            return caption
    return "No description available"


def generate_notes(slides, image_descriptions, output_path="converted_notes_gen.txt"):
    """Generate text notes using captions from the CSV"""
    notes = ""
    for slide in slides:
        notes += f"\n---\n## Slide {slide['slide_number']}: {slide['title']}\n\n"
        notes += f"**Content:**\n{slide['content']}\n\n"

        for img_path in slide['images']:
            caption = get_caption_from_descriptions(img_path, image_descriptions)
            notes += f"**Image:** {img_path}\nCaption: {caption}\n\n"

    with open(output_path, "w") as f:
        f.write(notes)
    print(f"✅ Notes saved to {output_path}")
    return output_path


def generate_pdf_notes(slides, image_descriptions, output_path="quick_notes_gen.pdf"):
    """Generate PDF notes using captions from the CSV"""
    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    margin = 2 * cm
    max_width = width - 2 * margin
    y_pos = height - margin

    for slide in slides:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y_pos, f"Slide {slide['slide_number']}: {slide['title']}")
        y_pos -= 1.2 * cm

        c.setFont("Helvetica", 11)
        for line in slide['content'].split("\n"):
            if y_pos < margin + 3 * cm:
                c.showPage()
                y_pos = height - margin
                c.setFont("Helvetica", 11)
            c.drawString(margin, y_pos, line.strip())
            y_pos -= 0.6 * cm

        for img_path in slide['images']:
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size
                scale = min(max_width / img_width, 8 * cm / img_height)
                img_width *= scale
                img_height *= scale

                if y_pos - img_height < margin:
                    c.showPage()
                    y_pos = height - margin

                c.drawImage(ImageReader(img), margin, y_pos - img_height, width=img_width, height=img_height)
                y_pos -= img_height + 0.5 * cm

                caption = get_caption_from_descriptions(img_path, image_descriptions)
                c.setFont("Helvetica-Oblique", 10)
                c.drawString(margin, y_pos, f"Caption: {caption}")
                y_pos -= 1.2 * cm
            except Exception as e:
                print(f"Skipping image {img_path}: {e}")

        c.showPage()
        y_pos = height - margin

    c.save()
    print(f"✅ PDF notes saved to: {output_path}")
    return output_path


def generate_image_descriptions_csv(slides, output_csv_path):
    """Generate CSV with image descriptions using DistilBART"""
    image_descriptions = []
    
    for slide in slides:
        slide_text = slide['content'][:500]  # Use first 500 chars of slide content for context
        
        for img_path in slide['images']:
            if '_0.' in img_path:  # Skip _0 images as in original captioning.py
                continue
                
            caption = generate_caption_with_distilbart(slide_text)
            image_descriptions.append((img_path, caption))
    
    # Save as CSV
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image Name', 'Description'])
        writer.writerows(image_descriptions)
    
    print(f"✅ CSV saved as {output_csv_path}")
    return image_descriptions


def process_pdf(pdf_path, output_text=None, output_pdf=None):
    """Process a single PDF file to extract content and generate notes"""
    print(f"Processing: {pdf_path}")
    slides = extract_pdf_content_filtered(pdf_path)
    
    # Generate base filename for outputs
    pdf_path_obj = Path(pdf_path)
    base_name = pdf_path_obj.stem
    output_dir = pdf_path_obj.parent
    
    # Generate CSV with image captions
    csv_path = output_dir / f"{base_name}_captions.csv"
    image_descriptions = generate_image_descriptions_csv(slides, csv_path)
    
    # Set default output paths if not specified
    if output_text is None:
        output_text = output_dir / f"{base_name}_gen.txt"
    elif output_text == "converted_notes.txt":  # Default value from argparse
        output_text = output_dir / f"{base_name}_gen.txt"
        
    if output_pdf is None:
        output_pdf = output_dir / f"{base_name}_gen.pdf"
    elif output_pdf == "quick_notes.pdf":  # Default value from argparse
        output_pdf = output_dir / f"{base_name}_gen.pdf"
    
    # Generate text notes
    generate_notes(slides, image_descriptions, output_path=output_text)
    
    # Generate PDF
    generate_pdf_notes(slides, image_descriptions, output_path=output_pdf)
    
    return slides, image_descriptions


def batch_extract_pdfs(dataset_dir="LLM_DATASET"):
    """Process all PDF files in the dataset directory with the new naming scheme"""
    dataset_path = Path(dataset_dir)
    pdf_files = list(dataset_path.rglob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf_file in pdf_files:
        # Skip files that have already been processed (contain "_cleaned" or "_gen" in the name)
        if "_cleaned" in pdf_file.stem or "_gen" in pdf_file.stem:
            continue
            
        base_name = pdf_file.stem
        output_txt = pdf_file.parent / f"{base_name}_gen.txt"
        output_pdf = pdf_file.parent / f"{base_name}_gen.pdf"
        process_pdf(pdf_file, output_text=output_txt, output_pdf=output_pdf)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract notes from PDF slides with DistilBART image captions.")
    parser.add_argument("pdf_path", nargs="?", help="Path to the PDF file")
    parser.add_argument("--output-text", help="Text output file (defaults to <filename>_gen.txt)")
    parser.add_argument("--output-pdf", help="PDF output file (defaults to <filename>_gen.pdf)")
    parser.add_argument("--batch", action="store_true", help="Run in batch mode on all PDFs in dataset directory")
    parser.add_argument("--dataset-dir", default="LLM_DATASET", help="Directory containing PDFs (used in batch mode)")
    args = parser.parse_args()

    if args.batch:
        batch_extract_pdfs(args.dataset_dir)
    elif args.pdf_path:
        process_pdf(args.pdf_path, args.output_text, args.output_pdf)
    else:
        print("❌ Please specify either a PDF path or use --batch for dataset-wide processing.")
