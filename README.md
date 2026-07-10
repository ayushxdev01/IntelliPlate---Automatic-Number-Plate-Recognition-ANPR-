# IntelliPlate

AI-powered Automatic Number Plate Recognition (ANPR) and traffic analytics platform for Indian vehicle plates. Built with FastAPI, OpenCV, Tesseract OCR, and a pretrained YOLOv8 vehicle classifier.

## Features

- **Live webcam feed** — captures a frame every 2 seconds and scans it for plates
- **Image upload** — scan a single photo directly
- **Video upload** — samples frames from an uploaded video and scans each one
- **Vehicle type classification** — Two-Wheeler / Four-Wheeler (Car/Bus/Truck), via a pretrained YOLOv8n (COCO) model
- **OCR confidence scoring** — reports Tesseract's confidence per detection, or `N/A` when the engine can't produce a reliable score, rather than showing a misleading `0`
- **Format-aware correction** — validates and auto-corrects OCR text against the Indian plate format (`SS DD LLL DDDD`), fixing common single/double-character OCR confusions (`O↔0`, `I↔1`, `S↔5/9`, `B↔8`, `Z↔2`, `G↔6`)
- **Ambiguity-safe** — if a correction could resolve to more than one equally valid plate, the system reports it as "needs review" instead of confidently guessing wrong
- **Duplicate detection** — groups near-identical OCR reads across video frames (via string similarity + per-character majority voting) so one physical vehicle isn't logged dozens of times
- **Persistent history** — SQLite-backed detection log with saved crop images
- **Dashboard** — searchable history table, daily stats (vehicles today, unique vehicles, peak hour, most frequent vehicle)
- **CSV export** — full detection log with date, time, plate, vehicle type, confidence, source, and a link to the saved image

## Tech Stack

| Layer | Tech |
|---|---|
| Backend | FastAPI, Uvicorn |
| Computer Vision | OpenCV (contour detection, CLAHE contrast enhancement, perspective transform) |
| OCR | Tesseract (via pytesseract) |
| Vehicle classification | YOLOv8n (Ultralytics), pretrained on COCO |
| Storage | SQLite, local filesystem for images |
| Frontend | Vanilla JS, Tailwind CSS |

## How plate detection works

1. Convert frame to grayscale (both as-is and with CLAHE contrast enhancement — CLAHE especially helps on dark-colored vehicles)
2. Canny edge detection → find contours → keep 4-corner shapes with a plate-like aspect ratio (2.0–6.0)
3. Perspective-warp the candidate region flat, threshold it (Otsu), and run Tesseract with a whitelist of A-Z0-9
4. Validate/correct the OCR text against the Indian plate regex, preferring a 2-digit-RTO reading over a 1-digit one when both are reachable (since real RTO codes are virtually always 2 digits)
5. If a detection is genuinely ambiguous (multiple equally valid corrections) or doesn't match the format at all, it's surfaced as "needs review" rather than silently dropped or guessed

## Known limitations (honest, not hidden)

This is a from-scratch contour+Tesseract pipeline, not a trained plate-detection model, and it has real limits worth knowing:

- **Best suited to fixed-angle camera footage** (e.g. a gate/entry camera at a consistent angle and distance) — it was tuned and validated against that kind of input. Close-up or steeply-angled phone photos can fail to detect a plate at all, or produce ambiguous readings.
- **No portrait/vertically-mounted plates** — the aspect-ratio filter assumes a landscape-oriented plate.
- **OCR confidence isn't always available** — depending on image quality, Tesseract sometimes doesn't return a usable per-character confidence; the app reports `N/A` in that case instead of a fake `0%`.
- **Not a substitute for a trained detector.** A proper object-detection model trained specifically on license plates (or a more robust OCR pipeline like EasyOCR) would generalize significantly better across angles/lighting than this contour-based approach. This is the natural next step if broader real-world robustness is needed.

## Possible next steps

- Swap the contour-based localizer for a trained plate-detection model (tested but not yet integrated: EasyOCR, which has its own scene-text detector)
- Blacklist/whitelist alerting (email/webhook) for security use cases
- Multi-camera support with per-camera labeling
- Authenticated admin panel for managing detection history and settings

## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Requires Tesseract OCR installed separately (not a pip package) — see [Tesseract's install docs](https://github.com/tesseract-ocr/tesseract) for your OS. On Windows, update the path check in `main.py` if installed somewhere other than `C:\Program Files\Tesseract-OCR\tesseract.exe`.

## Developed by

[Ayush Gupta](https://github.com/ayushxdev01)
