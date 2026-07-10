import cv2
import numpy as np
import pytesseract
import csv
import os
import io
import re
import sqlite3
import shutil
import difflib
import uuid
from collections import Counter
from datetime import datetime, date
from fastapi import FastAPI, File, UploadFile, Request, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image

try:
    from ultralytics import YOLO
    VEHICLE_MODEL = YOLO("yolov8n.pt")  # auto-downloads pretrained weights on first run
    VEHICLE_MODEL_AVAILABLE = True
except Exception as _e:
    VEHICLE_MODEL = None
    VEHICLE_MODEL_AVAILABLE = False
    print(f"[IntelliPlate] Vehicle-type model not loaded ({_e}). "
          f"vehicle_type will be reported as 'Unknown'.")

# COCO class ids -> vehicle type bucket. Only genuinely detectable classes
# are mapped here. Three-wheelers (auto-rickshaws) are NOT a COCO class,
# so we do not guess at them - anything that doesn't match falls back to
# "Unknown" rather than a fake label.
COCO_VEHICLE_MAP = {
    1: "Two-Wheeler (Bicycle)",
    3: "Two-Wheeler (Motorcycle)",
    2: "Four-Wheeler (Car)",
    5: "Four-Wheeler (Bus)",
    7: "Four-Wheeler (Truck)",
}

app = FastAPI(
    title="IntelliPlate",
    description="AI-Powered Automatic Number Plate Recognition and Traffic Analytics Platform",
)
templates = Jinja2Templates(directory="templates")

if os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

DB_FILE = "intelliplate.db"
IMAGES_DIR = "detections"
os.makedirs(IMAGES_DIR, exist_ok=True)
app.mount("/detections/image", StaticFiles(directory=IMAGES_DIR), name="detection_images")

DEBUG_DIR = "debug_output"
os.makedirs(DEBUG_DIR, exist_ok=True)
app.mount("/debug-frames", StaticFiles(directory=DEBUG_DIR, html=True), name="debug_frames")


# ---------------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            plate TEXT NOT NULL,
            confidence REAL,
            source TEXT,
            image_path TEXT,
            operator TEXT,
            vehicle_type TEXT
        )
    """)
    # For DBs created before vehicle_type existed - add the column if missing.
    existing_cols = [row["name"] for row in conn.execute("PRAGMA table_info(detections)").fetchall()]
    if "vehicle_type" not in existing_cols:
        conn.execute("ALTER TABLE detections ADD COLUMN vehicle_type TEXT")
    conn.commit()
    conn.close()


init_db()

print("=" * 60)
print("[IntelliPlate] Running main.py version: 2026-07-08-v4")
print(f"[IntelliPlate] Vehicle-type model loaded: {VEHICLE_MODEL_AVAILABLE}")
print("=" * 60)


# ---------------------------------------------------------------------------
# INDIAN PLATE FORMAT VALIDATION
# ---------------------------------------------------------------------------
# Standard format: 2 letters (state) + 2 digits (RTO code) + 1-3 letters
# (series) + 4 digits (unique number). Indian RTO codes are virtually
# always exactly 2 digits (01-99) - this STRICT regex is tried first.
STRICT_PLATE_REGEX = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$')

# Permissive fallback (1-2 digit RTO, 0-3 letter series) for the rare
# older/non-standard plates that don't fit the strict format - only used
# if NO correction satisfies the strict format above.
PLATE_REGEX = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$')

# Characters Tesseract commonly confuses with each other on plates
# (based on the real misreads seen in your session logs).
AMBIGUOUS_MAP = {
    '0': ['O', 'D', 'Q'],
    'O': ['0', 'D', 'Q'],
    'D': ['0', 'O'],
    '1': ['I'],
    'I': ['1'],
    '5': ['S'],
    'S': ['5', '9'],
    '8': ['B'],
    'B': ['8'],
    '2': ['Z'],
    'Z': ['2'],
    '6': ['G'],
    'G': ['6'],
    '9': ['D', 'S'],
}


def _try_corrections(text, regex):
    """Tries 0, then 1, then 2 ambiguous-character substitutions against
    the given regex. Returns (result, is_ambiguous):
      - (match, False)      if exactly one match is found at the winning tier
      - (first_match, True) if MULTIPLE different matches are found at the
                             same tier - i.e. genuinely ambiguous, we can't
                             tell which is correct without more signal
      - (None, False)       if nothing matches at all
    """
    if regex.match(text):
        return text, False

    ambiguous_positions = [i for i, c in enumerate(text) if c in AMBIGUOUS_MAP]

    tier1_matches = []
    for i in ambiguous_positions:
        for repl in AMBIGUOUS_MAP[text[i]]:
            candidate = text[:i] + repl + text[i + 1:]
            if regex.match(candidate) and candidate not in tier1_matches:
                tier1_matches.append(candidate)
    if tier1_matches:
        return tier1_matches[0], len(tier1_matches) > 1

    tier2_matches = []
    for a_idx, i in enumerate(ambiguous_positions[:6]):
        for j in ambiguous_positions[a_idx + 1:6]:
            for repl_i in AMBIGUOUS_MAP[text[i]]:
                for repl_j in AMBIGUOUS_MAP[text[j]]:
                    candidate = list(text)
                    candidate[i] = repl_i
                    candidate[j] = repl_j
                    candidate = "".join(candidate)
                    if regex.match(candidate) and candidate not in tier2_matches:
                        tier2_matches.append(candidate)
    if tier2_matches:
        return tier2_matches[0], len(tier2_matches) > 1

    return None, False


def correct_and_validate(text):
    """
    Turns a raw OCR string into a string matching the Indian plate format,
    if possible. Returns None if nothing valid could be produced (i.e.
    this probably isn't a real plate at all, e.g. "SELECT", "EEEE") OR if
    multiple equally-plausible corrections exist and we have no reliable
    way to pick between them (better to say "not sure" than guess wrong).

    Tries the STRICT (2-digit RTO) format first, across 0/1/2 character
    corrections, before falling back to the permissive format. This
    matters because a "cheaper" 1-edit fix can produce a plausible-but-
    WRONG plate (e.g. "TN0SBY9726") while the CORRECT plate needs 2 edits
    ("TN09BY9726") - without this priority, the cheap wrong answer would
    win just because it took fewer edits to reach.
    """
    text = text.upper()

    strict_result, strict_ambiguous = _try_corrections(text, STRICT_PLATE_REGEX)
    if strict_result and not strict_ambiguous:
        return strict_result
    if strict_result and strict_ambiguous:
        return None  # multiple valid strict-format readings - don't guess

    permissive_result, permissive_ambiguous = _try_corrections(text, PLATE_REGEX)
    if permissive_result and not permissive_ambiguous:
        return permissive_result
    return None


def save_detection(plate_text, confidence, source, crop_image=None, vehicle_type="Unknown", operator="Ayush Gupta"):
    """Saves one detection row to SQLite, and the plate crop image to disk."""
    image_path = None
    if crop_image is not None:
        filename = f"{plate_text}_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(IMAGES_DIR, filename)
        cv2.imwrite(filepath, crop_image)
        image_path = filename  # store relative filename; served via /detections/image/<filename>

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    conn.execute(
        "INSERT INTO detections (timestamp, plate, confidence, source, image_path, operator, vehicle_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (timestamp, plate_text, confidence, source, image_path, operator, vehicle_type),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# IMAGE PROCESSING / OCR
# ---------------------------------------------------------------------------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_perspective_transform(image, pts):
    rect = order_points(pts.reshape(4, 2))
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((br[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def detect_vehicle_type(frame):
    """
    Runs a pretrained YOLOv8 (COCO) model on the full frame and returns the
    vehicle type of the highest-confidence detected vehicle, e.g.
    "Four-Wheeler (Car)", "Two-Wheeler (Motorcycle)". Returns "Unknown" if
    the model isn't available or no recognizable vehicle class is found.
    NOTE: brand/model name (e.g. "Maruti", "Honda") and three-wheelers are
    NOT supported - COCO has no such classes, and guessing would just be
    a fake label, not a real detection.
    """
    if not VEHICLE_MODEL_AVAILABLE:
        return "Unknown"

    try:
        results = VEHICLE_MODEL(frame, verbose=False)[0]
        best_label = "Unknown"
        best_conf = 0.0
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id in COCO_VEHICLE_MAP and conf > best_conf:
                best_conf = conf
                best_label = COCO_VEHICLE_MAP[cls_id]
        return best_label
    except Exception:
        return "Unknown"


def ocr_with_confidence(thresh_img):
    """Runs Tesseract and returns (text, average_confidence_0_to_100_or_None).
    Returns None for confidence (not 0.0) when Tesseract doesn't report any
    usable per-word confidence - 0.0 would falsely imply "detected but zero
    confidence", which is misleading when text was in fact recognized."""
    custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7'
    data = pytesseract.image_to_data(thresh_img, config=custom_config, output_type=pytesseract.Output.DICT)

    words = []
    confidences = []
    for word, conf in zip(data.get("text", []), data.get("conf", [])):
        if word.strip():
            words.append(word)
            try:
                c = float(conf)
                if c >= 0:
                    confidences.append(c)
            except (ValueError, TypeError):
                continue

    text = "".join(words)
    if not confidences:
        avg_conf = None
    elif all(c == 0 for c in confidences):
        # Tesseract reporting a flat 0.0 for every single word (across many
        # different plates/runs) is not a real "zero confidence" reading -
        # it's a sign the confidence engine itself isn't working properly
        # in this Tesseract install (common with older/incomplete builds).
        # Treat it as "unavailable" rather than a fake precise-looking 0.0.
        avg_conf = None
    else:
        avg_conf = round(sum(confidences) / len(confidences), 1)
    return text, avg_conf


def _find_plate_candidates(gray, debug_frame=None, variant_label=""):
    """Runs the contour-based plate search on a given grayscale variant
    and returns a list of candidate dicts.

    If debug_frame is given (a BGR image to draw on), every 4-point
    contour that was tried gets outlined on it:
      - RED    = rejected by aspect ratio (not plate-shaped)
      - YELLOW = right shape, but OCR text didn't pass length check
      - GREEN  = accepted as a candidate
    This lets you visually check whether a real plate's region was even
    considered at all, instead of guessing blind.
    """
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blur, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    candidates = []

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            if not (2.0 < aspect_ratio < 6.0):
                if debug_frame is not None:
                    cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(debug_frame, f"{variant_label} ar={aspect_ratio:.1f}", (x, max(y - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                continue

            try:
                warped_plate = get_perspective_transform(gray, approx)
                ch, cw = warped_plate.shape[:2]
                pad_h, pad_w = int(ch * 0.08), int(cw * 0.04)
                if ch - 2 * pad_h > 10 and cw - 2 * pad_w > 10:
                    warped_plate = warped_plate[pad_h:ch - pad_h, pad_w:cw - pad_w]

                warped_plate_resized = cv2.resize(warped_plate, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
                _, thresh = cv2.threshold(warped_plate_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                text, confidence = ocr_with_confidence(thresh)
                clean_text = "".join(e for e in text if e.isalnum()).upper()

                if 4 <= len(clean_text) <= 11:
                    candidates.append({
                        "text": clean_text,
                        "confidence": confidence,
                        "crop": warped_plate_resized,
                    })
                    if debug_frame is not None:
                        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                        cv2.putText(debug_frame, f"{variant_label} {clean_text}", (x, max(y - 5, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
                elif debug_frame is not None:
                    cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 220, 220), 2)
                    cv2.putText(debug_frame, f"{variant_label} txt='{text}'", (x, max(y - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 180), 1)
            except Exception:
                continue

    return candidates


def process_anpr_frame(frame, debug_frame=None):
    """
    Scans a frame for plate-shaped regions and returns the best candidate
    as a dict: {"text": raw_ocr_text, "confidence": float, "crop": image}
    or None if nothing found.

    Runs the search on both the plain grayscale image AND a CLAHE
    (contrast-enhanced) version, then picks the best result across both.
    This is because CLAHE helps on low-contrast/dark vehicles but can
    over-enhance noise/compression artifacts on some footage - running
    both avoids regressing on videos where the plain version worked fine.

    NOTE: A full-image sparse-text OCR fallback was tried here (to catch
    plates the contour detector misses) and deliberately removed - testing
    showed it sometimes matches random background text to the plate regex
    format and returns a confident WRONG answer, which is worse than
    honestly returning nothing. Real fix needs a proper trained
    plate-detector model, not a bigger regex net.

    If debug_frame is given, every contour tried on this frame gets drawn
    on it (red=wrong shape, yellow=right shape but OCR text rejected,
    green=accepted) so you can SEE what the detector is doing instead of
    guessing.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    candidates = (
        _find_plate_candidates(gray, debug_frame, "plain")
        + _find_plate_candidates(gray_enhanced, debug_frame, "clahe")
    )

    if not candidates:
        return None

    # Prefer candidates whose text already matches (or can be corrected
    # to match) the plate format; among those, pick the highest confidence
    # (candidates with unknown/None confidence are treated as lowest
    # priority, not as zero-confidence).
    valid_candidates = [c for c in candidates if correct_and_validate(c["text"])]
    pool = valid_candidates if valid_candidates else candidates
    best = max(pool, key=lambda c: c["confidence"] if c["confidence"] is not None else -1)

    # Only run the (heavier) vehicle-type model once we know this frame
    # actually has a plate-shaped region worth acting on.
    best["vehicle_type"] = detect_vehicle_type(frame)
    return best


def group_and_vote(raw_reads):
    """
    Groups near-duplicate OCR reads of what is probably the SAME physical
    plate (across frames of a video) and returns one entry per group:
    {"text": voted_text, "confidence": avg_confidence, "crop": best_crop}
    This stops one plate being logged as many slightly different strings.
    """
    groups = []

    for read in raw_reads:
        placed = False
        for group in groups:
            similarity = difflib.SequenceMatcher(None, read["text"], group[0]["text"]).ratio()
            if similarity >= 0.75:
                group.append(read)
                placed = True
                break
        if not placed:
            groups.append([read])

    results = []
    for group in groups:
        lengths = Counter(len(r["text"]) for r in group)
        target_len = lengths.most_common(1)[0][0]
        same_len = [r for r in group if len(r["text"]) == target_len]

        voted_chars = []
        for i in range(target_len):
            counts = Counter(r["text"][i] for r in same_len)
            voted_chars.append(counts.most_common(1)[0][0])
        voted_text = "".join(voted_chars)

        known_confidences = [r["confidence"] for r in group if r["confidence"] is not None]
        avg_confidence = round(sum(known_confidences) / len(known_confidences), 1) if known_confidences else None
        best_crop = max(group, key=lambda r: r["confidence"] if r["confidence"] is not None else -1)["crop"]
        vehicle_type = Counter(r.get("vehicle_type", "Unknown") for r in group).most_common(1)[0][0]

        results.append({
            "text": voted_text,
            "confidence": avg_confidence,
            "crop": best_crop,
            "vehicle_type": vehicle_type,
        })

    return results


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/scan/")
async def scan_uploaded_frame(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        result = process_anpr_frame(frame)
        if not result:
            return {"success": False, "message": "No plate detected"}

        corrected = correct_and_validate(result["text"])
        if not corrected:
            return {
                "success": False,
                "message": "Detected region did not match plate format",
                "raw_ocr_text": result["text"],
                "confidence": result["confidence"],
            }

        save_detection(
            corrected, result["confidence"], source="image",
            crop_image=result["crop"], vehicle_type=result.get("vehicle_type", "Unknown"),
        )
        return {
            "success": True,
            "plate_number": corrected,
            "confidence": result["confidence"],
            "raw_ocr_text": result["text"],
            "vehicle_type": result.get("vehicle_type", "Unknown"),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...), debug: bool = Query(False)):
    temp_video_path = f"temp_{file.filename}"
    debug_dir = "debug_output"
    try:
        if debug:
            if os.path.exists(debug_dir):
                shutil.rmtree(debug_dir)
            os.makedirs(debug_dir, exist_ok=True)

        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        cap = cv2.VideoCapture(temp_video_path)
        raw_reads = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Sample every 3rd frame instead of every 8th - catches plates
            # that were only clearly visible for a short window (e.g. a
            # car passing through frame quickly), at the cost of somewhat
            # slower processing.
            if frame_count % 3 == 0:
                debug_frame = frame.copy() if debug else None
                result = process_anpr_frame(frame, debug_frame=debug_frame)
                if result:
                    raw_reads.append(result)

                if debug:
                    # RED = wrong shape, YELLOW = right shape but OCR text
                    # rejected, GREEN = accepted as a candidate.
                    cv2.imwrite(os.path.join(debug_dir, f"frame_{frame_count:05d}.jpg"), debug_frame)

            frame_count += 1

        cap.release()
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        voted = group_and_vote(raw_reads)

        detected_plates = []
        needs_review = []  # candidates that were found but didn't pass
                            # strict format validation - NOT silently
                            # dropped, so nothing gets lost like the Volvo
                            # plate did before this fix.
        seen_plates = set()
        for entry in voted:
            corrected = correct_and_validate(entry["text"])
            if corrected and corrected not in seen_plates:
                seen_plates.add(corrected)
                save_detection(
                    corrected, entry["confidence"], source="video",
                    crop_image=entry["crop"], vehicle_type=entry.get("vehicle_type", "Unknown"),
                )
                detected_plates.append({
                    "plate": corrected,
                    "confidence": entry["confidence"],
                    "vehicle_type": entry.get("vehicle_type", "Unknown"),
                })
            elif not corrected:
                needs_review.append({
                    "raw_ocr_text": entry["text"],
                    "confidence": entry["confidence"],
                    "vehicle_type": entry.get("vehicle_type", "Unknown"),
                    "reason": "Detected a plate-shaped region but the OCR text didn't match the Indian plate format even after correction.",
                })

        return {
            "success": True,
            "detected_plates": detected_plates,
            "needs_review": needs_review,
        }
    except Exception as e:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return {"success": False, "error": str(e)}


@app.get("/history")
async def get_history(
    search: str = Query(None, description="Filter by plate number (partial match)"),
    start_date: str = Query(None, description="YYYY-MM-DD"),
    end_date: str = Query(None, description="YYYY-MM-DD"),
    source: str = Query(None, description="Filter by source: image / video"),
    limit: int = Query(50, le=500),
    offset: int = Query(0),
):
    conn = get_db()
    query = "SELECT * FROM detections WHERE 1=1"
    params = []

    if search:
        query += " AND plate LIKE ?"
        params.append(f"%{search.upper()}%")
    if start_date:
        query += " AND date(timestamp) >= date(?)"
        params.append(start_date)
    if end_date:
        query += " AND date(timestamp) <= date(?)"
        params.append(end_date)
    if source:
        query += " AND source = ?"
        params.append(source)

    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(query, params).fetchall()
    conn.close()

    results = []
    for row in rows:
        image_url = f"/detections/image/{row['image_path']}" if row["image_path"] else None
        results.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "plate": row["plate"],
            "confidence": row["confidence"],
            "source": row["source"],
            "vehicle_type": row["vehicle_type"],
            "image_url": image_url,
        })

    return {"success": True, "count": len(results), "results": results}


@app.get("/vehicle/{plate}")
async def get_vehicle_history(plate: str):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM detections WHERE plate = ? ORDER BY timestamp DESC",
        (plate.upper(),),
    ).fetchall()
    conn.close()

    if not rows:
        return JSONResponse(status_code=404, content={"success": False, "message": "No detections found for this plate"})

    results = [{
        "id": row["id"],
        "timestamp": row["timestamp"],
        "confidence": row["confidence"],
        "source": row["source"],
        "vehicle_type": row["vehicle_type"],
        "image_url": f"/detections/image/{row['image_path']}" if row["image_path"] else None,
    } for row in rows]

    return {"success": True, "plate": plate.upper(), "total_detections": len(results), "history": results}


@app.get("/debug/tesseract-info")
async def debug_tesseract_info():
    """
    Diagnostic endpoint - reports exactly which Tesseract binary this app
    is using, its version, and what raw confidence values it returns on a
    simple test image. Use this instead of guessing why OCR confidence
    looks wrong.
    """
    info = {}

    try:
        info["tesseract_cmd_path"] = pytesseract.pytesseract.tesseract_cmd
    except Exception as e:
        info["tesseract_cmd_path"] = f"ERROR: {e}"

    try:
        info["tesseract_version"] = str(pytesseract.get_tesseract_version())
    except Exception as e:
        info["tesseract_version"] = f"ERROR: {e}"

    # Build a simple black-text-on-white test image and OCR it, so we can
    # see the RAW confidence values Tesseract reports in this environment.
    try:
        test_img = np.full((60, 220), 255, dtype=np.uint8)
        cv2.putText(test_img, "AB1234", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,), 2)
        custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 7'
        data = pytesseract.image_to_data(test_img, config=custom_config, output_type=pytesseract.Output.DICT)
        info["test_image_raw_words"] = data.get("text", [])
        info["test_image_raw_confidences"] = data.get("conf", [])
    except Exception as e:
        info["test_image_error"] = f"ERROR: {e}"

    return info


@app.get("/analytics")
async def get_analytics():
    conn = get_db()
    today_str = date.today().strftime("%Y-%m-%d")

    total_today = conn.execute(
        "SELECT COUNT(*) as c FROM detections WHERE date(timestamp) = date(?)", (today_str,)
    ).fetchone()["c"]

    unique_today = conn.execute(
        "SELECT COUNT(DISTINCT plate) as c FROM detections WHERE date(timestamp) = date(?)", (today_str,)
    ).fetchone()["c"]

    avg_conf_row = conn.execute("SELECT AVG(confidence) as avg_conf FROM detections").fetchone()
    avg_confidence = round(avg_conf_row["avg_conf"], 1) if avg_conf_row["avg_conf"] is not None else 0.0

    most_frequent_row = conn.execute(
        "SELECT plate, COUNT(*) as cnt FROM detections GROUP BY plate ORDER BY cnt DESC LIMIT 1"
    ).fetchone()
    most_frequent = {"plate": most_frequent_row["plate"], "count": most_frequent_row["cnt"]} if most_frequent_row else None

    peak_hour_row = conn.execute(
        "SELECT strftime('%H', timestamp) as hour, COUNT(*) as cnt FROM detections "
        "GROUP BY hour ORDER BY cnt DESC LIMIT 1"
    ).fetchone()
    peak_hour = f"{peak_hour_row['hour']}:00" if peak_hour_row and peak_hour_row["hour"] is not None else None

    total_all_time = conn.execute("SELECT COUNT(*) as c FROM detections").fetchone()["c"]
    unique_all_time = conn.execute("SELECT COUNT(DISTINCT plate) as c FROM detections").fetchone()["c"]

    conn.close()

    return {
        "success": True,
        "vehicles_today": total_today,
        "unique_vehicles_today": unique_today,
        "total_detections_all_time": total_all_time,
        "unique_vehicles_all_time": unique_all_time,
        "average_ocr_confidence": avg_confidence,
        "most_frequent_vehicle": most_frequent,
        "peak_hour": peak_hour,
        # NOTE: "detection accuracy" is intentionally NOT reported here -
        # that requires ground-truth labelled data to compute honestly.
        # average_ocr_confidence is the honest proxy metric we can report.
    }


@app.get("/export-csv")
async def export_csv(request: Request):
    conn = get_db()
    rows = conn.execute(
        "SELECT timestamp, plate, confidence, source, vehicle_type, image_path, operator "
        "FROM detections ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()

    base_url = str(request.base_url).rstrip("/")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Date", "Time", "License Plate", "Vehicle Type",
        "OCR Confidence (%)", "Source", "Image URL", "Operator",
    ])
    for row in rows:
        date_part, time_part = row["timestamp"].split(" ") if " " in row["timestamp"] else (row["timestamp"], "")
        image_url = f"{base_url}/detections/image/{row['image_path']}" if row["image_path"] else ""
        confidence_display = row["confidence"] if row["confidence"] is not None else "N/A"
        writer.writerow([
            date_part,
            time_part,
            row["plate"],
            row["vehicle_type"] or "Unknown",
            confidence_display,
            row["source"],
            image_url,
            row["operator"],
        ])

    output.seek(0)
    filename = f"intelliplate_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.delete("/records/all")
async def delete_all_records():
    """Deletes every detection record AND every saved plate-crop image.
    Irreversible - the frontend requires the user to type a confirmation
    phrase before calling this."""
    conn = get_db()
    rows = conn.execute("SELECT image_path FROM detections").fetchall()

    for row in rows:
        if row["image_path"]:
            image_file = os.path.join(IMAGES_DIR, row["image_path"])
            if os.path.exists(image_file):
                os.remove(image_file)

    deleted_count = conn.execute("SELECT COUNT(*) as c FROM detections").fetchone()["c"]
    conn.execute("DELETE FROM detections")
    conn.commit()
    conn.close()

    return {"success": True, "message": f"Deleted {deleted_count} record(s)"}


@app.delete("/record/{record_id}")
async def delete_record(record_id: int):
    conn = get_db()
    row = conn.execute("SELECT image_path FROM detections WHERE id = ?", (record_id,)).fetchone()
    if not row:
        conn.close()
        return JSONResponse(status_code=404, content={"success": False, "message": "Record not found"})

    if row["image_path"]:
        image_file = os.path.join(IMAGES_DIR, row["image_path"])
        if os.path.exists(image_file):
            os.remove(image_file)

    conn.execute("DELETE FROM detections WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return {"success": True, "message": f"Record {record_id} deleted"}