FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y libreoffice && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

CMD ["python", "pptx_to_pdf.py"]
