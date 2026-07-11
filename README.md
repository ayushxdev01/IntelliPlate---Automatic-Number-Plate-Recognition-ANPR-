<div align="center">

# 🚗 IntelliPlate

**AI-Powered Automatic Number Plate Recognition & Traffic Analytics Platform**

Built for Indian vehicle plates — live webcam, image, and video ANPR with a real-time analytics dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-5C3EE8?logo=opencv&logoColor=white)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-4285F4)
![YOLOv8](https://img.shields.io/badge/YOLOv8-VehicleDetection-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

[Features](#-features) • [Tech Stack](#-tech-stack) • [How It Works](#-how-plate-detection-works) • [Limitations](#-known-limitations) • [Setup](#-setup) • [Roadmap](#-possible-next-steps)

</div>

---

## 📸 Demo

<div align="center">
<i>Add a screenshot or GIF of the dashboard here — recruiters/visitors judge a project by this section first.</i>

<!-- ![IntelliPlate Dashboard](docs/demo.gif) -->

**🔗 Live Demo:** [intelliplate-automatic-number-plate.onrender.com](https://intelliplate-automatic-number-plate.onrender.com/)

> Note: hosted on Render's free tier — the app may take 30-60 seconds to wake up on the first request after a period of inactivity (cold start).

</div>

---

## ✨ Features

| | |
|---|---|
| 📹 **Live webcam feed** | Captures a frame every 2 seconds and scans it for plates |
| 🖼️ **Image upload** | Scan a single photo directly |
| 🎞️ **Video upload** | Samples frames from an uploaded video and scans each one |
| 🚙 **Vehicle type classification** | Two-Wheeler / Four-Wheeler (Car/Bus/Truck) via a pretrained YOLOv8n (COCO) model |
| 📊 **OCR confidence scoring** | Reports Tesseract's real confidence, or `N/A` when unavailable — never a misleading fake `0` |
| 🔤 **Format-aware correction** | Validates/auto-corrects OCR text against the Indian plate format, fixing common OCR confusions (`O↔0`, `I↔1`, `S↔5/9`, `B↔8`, `Z↔2`, `G↔6`) |
| ⚠️ **Ambiguity-safe** | If a correction could resolve to more than one equally valid plate, it's flagged "needs review" instead of confidently guessing wrong |
| 🔁 **Duplicate detection** | Groups near-identical OCR reads across video frames (similarity + majority voting) so one vehicle isn't logged dozens of times |
| 🗄️ **Persistent history** | SQLite-backed detection log with saved crop images |
| 📈 **Dashboard** | Searchable history, daily stats — vehicles today, unique vehicles, peak hour, most frequent vehicle |
| 📤 **CSV export** | Full detection log: date, time, plate, vehicle type, confidence, source, image link |
| 🗑️ **Record management** | Delete individual detections or clear all history from the dashboard |

---

## 🛠 Tech Stack

| Layer | Tech |
|---|---|
| Backend | FastAPI, Uvicorn |
| Computer Vision | OpenCV — contour detection, CLAHE contrast enhancement, perspective transform |
| OCR | Tesseract (via pytesseract) |
| Vehicle classification | YOLOv8n (Ultralytics), pretrained on COCO |
| Storage | SQLite + local filesystem for images |
| Frontend | Vanilla JS, Tailwind CSS |
| Deployment | Docker |

---

## 🔍 How plate detection works

```
Frame → Grayscale (plain + CLAHE-enhanced) → Canny edges → Contours
      → Keep 4-corner shapes with plate-like aspect ratio (2.0–6.0)
      → Perspective-warp flat → Otsu threshold → Tesseract OCR
      → Validate/correct against Indian plate format → Save or flag for review
```

1. Convert frame to grayscale — both as-is and CLAHE-enhanced (helps on dark-colored vehicles)
2. Canny edge detection → find contours → keep 4-corner shapes with a plate-like aspect ratio
3. Perspective-warp the candidate region flat, threshold it (Otsu), run Tesseract with an A-Z0-9 whitelist
4. Validate/correct the text against the Indian plate regex — preferring a 2-digit-RTO reading over a 1-digit one when both are reachable, since real RTO codes are virtually always 2 digits
5. If a detection is genuinely ambiguous, or doesn't match the format at all, it's surfaced as **needs review** rather than silently dropped or guessed

---

## ⚠️ Known Limitations

<details>
<summary><b>Click to expand — honest limitations, not hidden</b></summary>
<br>

This is a from-scratch contour + Tesseract pipeline, not a trained plate-detection model, and it has real limits worth knowing:

- **Best suited to fixed-angle camera footage** (e.g. a gate/entry camera at a consistent angle and distance) — tuned and validated against that kind of input. Close-up or steeply-angled phone photos can fail to detect a plate at all, or produce ambiguous readings.
- **No portrait/vertically-mounted plates** — the aspect-ratio filter assumes a landscape-oriented plate.
- **OCR confidence isn't always available** — depending on image quality, Tesseract sometimes can't return a usable per-character confidence; the app reports `N/A` rather than faking a `0%`.
- **Not a substitute for a trained detector.** A proper object-detection model trained specifically on license plates (or a more robust OCR pipeline like EasyOCR) would generalize significantly better across angles/lighting than this contour-based approach — the natural next step for broader real-world robustness.

</details>

---

## 🗺 Possible next steps

- [ ] Swap the contour-based localizer for a trained plate-detection model (EasyOCR's scene-text detector was evaluated but not yet integrated)
- [ ] Blacklist/whitelist alerting (email/webhook) for security use cases
- [ ] Multi-camera support with per-camera labeling
- [ ] Authenticated admin panel for managing detection history and settings

---

## 🚀 Setup

<details>
<summary><b>Local setup</b></summary>
<br>

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Requires **Tesseract OCR** installed separately (it's a system binary, not a pip package) — see the [Tesseract install docs](https://github.com/tesseract-ocr/tesseract) for your OS. On Windows, update the path check in `main.py` if installed somewhere other than `C:\Program Files\Tesseract-OCR\tesseract.exe`.

</details>

<details>
<summary><b>Docker</b></summary>
<br>

```bash
docker build -t intelliplate .
docker run -p 8000:8000 intelliplate
```

The included `Dockerfile` installs Tesseract as a system package, so no separate install is needed in this path.

</details>

---

## 👤 Developed by

**[Ayush Gupta](https://github.com/ayushxdev01)**
