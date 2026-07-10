FROM python:3.11-slim

# Tesseract OCR is a system binary, not a pip package - install it here.
# libgl1/libglib2.0-0 are needed by opencv-python-headless at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# On Linux, tesseract is on PATH after apt-get install, so pytesseract
# finds it automatically - no manual tesseract_cmd path needed like on
# Windows (see the os.path.exists check for the Windows path in main.py).

EXPOSE 8000

# Render/Railway set a PORT env var (Render defaults to 10000) and expect
# the app to bind to it - falls back to 8000 for local/plain `docker run`.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]