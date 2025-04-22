import os
import hashlib
import fitz  # PyMuPDF
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import cm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from pathlib import Path

# Lazy loading for BLIP-2 model
_blip2_model = None
_blip2_processor = None


def get_blip2():
    global _blip2_model, _blip2_processor
    if _blip2_model is None or _blip2_processor is None:
        _blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
        _blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").eval()
    return _blip2_model, _blip2_processor


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


def generate_contextual_description(image_path, slide_text):
    question = (
        "Based on the context of the slide, explain what this image is showing. "
        "Focus on describing any diagrams, tables, or flowcharts in relation to the slide content."
    )
    prompt = f"{slide_text}\n\n{question}"
    image = Image.open(image_path).convert('RGB')
    model, processor = get_blip2()
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=100)
    return processor.decode(out[0], skip_special_tokens=True)


def generate_notes(slides, output_path="converted_notes.txt"):
    notes = ""
    for slide in slides:
        notes += f"\n---\n## Slide {slide['slide_number']}: {slide['title']}\n\n"
        notes += f"**Content:**\n{slide['content']}\n\n"

        for img_path in slide['images']:
            caption = generate_contextual_description(img_path, slide['content'][:500])
            notes += f"**Image:** {img_path}\nCaption: {caption}\n\n"

    with open(output_path, "w") as f:
        f.write(notes)
    print(f"✅ Notes saved to {output_path}")
    return output_path


def generate_pdf_notes(slides, output_path="quick_notes.pdf"):
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

                caption = generate_contextual_description(img_path, slide['content'][:500])
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


def process_pdf(pdf_path, output_text=None, output_pdf=None):
    print(f"Processing: {pdf_path}")
    slides = extract_pdf_content_filtered(pdf_path)
    if output_text:
        generate_notes(slides, output_path=output_text)
    if output_pdf:
        generate_pdf_notes(slides, output_path=output_pdf)


def batch_extract_pdfs(dataset_dir="LLM_DATASET"):
    dataset_path = Path(dataset_dir)
    pdf_files = list(dataset_path.rglob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf_file in pdf_files:
        output_txt = pdf_file.with_suffix(".txt")
        output_pdf = pdf_file.with_name(pdf_file.stem + "_cleaned.pdf")
        process_pdf(pdf_file, output_text=output_txt, output_pdf=output_pdf)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract notes from PDF slides with contextual image captions.")
    parser.add_argument("pdf_path", nargs="?", help="Path to the PDF file")
    parser.add_argument("--output-text", default="converted_notes.txt", help="Text output file")
    parser.add_argument("--output-pdf", default="quick_notes.pdf", help="PDF output file")
    parser.add_argument("--batch", action="store_true", help="Run in batch mode on all PDFs in dataset directory")
    parser.add_argument("--dataset-dir", default="LLM_DATASET", help="Directory containing PDFs (used in batch mode)")
    args = parser.parse_args()

    if args.batch:
        batch_extract_pdfs(args.dataset_dir)
    elif args.pdf_path:
        process_pdf(args.pdf_path, args.output_text, args.output_pdf)
    else:
        print("❌ Please specify either a PDF path or use --batch for dataset-wide processing.")
