# backend/pipeline/pipeline_esri.py
import os
import math
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
import traceback

# import your existing fetch function (should be present as pipeline/fetch_tile_esri.py)
from pipeline.fetch_tile_esri import fetch_esri_tile

# ---------- CONFIG ----------
# Update this ONLY if your model is somewhere else
MODEL_PATH = os.path.abspath(r"C:/Users/arlen/OneDrive/Desktop/eco_ideathon/backend/runs/detect/solar_detector/weights/best.pt")

# output dir inside pipeline folder -> served by app.py as /outputs/<file>
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static", "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# tile request params
DEFAULT_ZOOM = 19        # try 19; lower to 18 if ESRI returns 'map data not available'
TILE_SIZE = 640          # produce 640x640 tile (YOLO-friendly)

# Load YOLO model (fail fast if missing)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"YOLO weights not found at: {MODEL_PATH}")
MODEL = YOLO(MODEL_PATH)


def _meters_per_pixel(lat, zoom):
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


def bbox_area_sqm(bbox, lat, zoom=DEFAULT_ZOOM):
    # bbox pixel area -> meters^2 using meters_per_pixel for provided zoom
    x1, y1, x2, y2 = bbox
    px_area = abs((x2 - x1) * (y2 - y1))
    mpp = _meters_per_pixel(lat, zoom)
    return px_area * (mpp ** 2)


def _safe_save_rgb(img_path):
    # ensure saved as RGB (YOLO + browser friendly)
    im = Image.open(img_path).convert("RGB")
    im.save(img_path)


def run_pipeline_single(sample_id: int, lat: float, lon: float, zoom=DEFAULT_ZOOM):
    """
    Fetch tile -> run YOLO -> save overlay -> return JSON record (mandatory format)
    """
    tile_path = os.path.join(OUTPUT_DIR, "tile.jpg")
    overlay_path = os.path.join(OUTPUT_DIR, "overlay.jpg")

    # 1) Fetch tile (ESRI; if ESRI fails, raise and let caller handle)
    try:
        fetch_esri_tile(lat, lon, tile_path, zoom=zoom, size=TILE_SIZE)
        source_name = "ESRI"
    except Exception as e_esri:
        # if ESRI fails, attempt a fallback (optional)
        # You can implement a fetch_google_tile function if you have it.
        # For now, return NOT_VERIFIABLE with tile_image set to whatever was written (if any)
        try:
            # if your repo has a fetch_tile_google, uncomment and use:
            # from pipeline.fetch_tile_google import fetch_google_tile
            # fetch_google_tile(lat, lon, tile_path, zoom=zoom, size=TILE_SIZE)
            # source_name = "GOOGLE_TILES_FALLBACK"
            # For safety here, re-raise and return an error JSON below
            raise e_esri
        except Exception:
            # return a NOT_VERIFIABLE response (tile may not exist)
            return {
                "sample_id": sample_id,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "has_solar": False,
                "confidence": 0,
                "pv_area_sqm_est": 0,
                "buffer_radius_sqft": 1200,
                "qc_status": "NOT_VERIFIABLE",
                "bbox_or_mask": None,
                "image_metadata": {"source": None, "capture_date": None},
                "tile_image": tile_path.replace("\\", "/") if os.path.exists(tile_path) else None,
                "overlay_image": None
            }

    # Ensure tile is RGB and saved
    try:
        _safe_save_rgb(tile_path)
    except Exception:
        pass

    # 2) Run YOLO inference; force saves into runs/detect/web_predict
    try:
        results = MODEL.predict(
            source=tile_path,
            conf=0.30,
            save=True,
            project=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs", "detect")),
            name="web_predict",
            exist_ok=True
        )
    except Exception as e:
        traceback.print_exc()
        # If inference fails, return NOT_VERIFIABLE result but keep tile path
        return {
            "sample_id": sample_id,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "has_solar": False,
            "confidence": 0,
            "pv_area_sqm_est": 0,
            "buffer_radius_sqft": 1200,
            "qc_status": "NOT_VERIFIABLE",
            "bbox_or_mask": None,
            "image_metadata": {"source": "ESRI", "capture_date": None},
            "tile_image": tile_path.replace("\\", "/"),
            "overlay_image": None
        }

    # find the saved overlay image in YOLO's predict folder
    pred_dir = results[0].save_dir
    saved_images = [f for f in os.listdir(pred_dir) if f.lower().endswith((".jpg", ".png"))]
    if len(saved_images) == 0:
        # YOLO saved nothing; return tile-only result
        return {
            "sample_id": sample_id,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "has_solar": False,
            "confidence": 0,
            "pv_area_sqm_est": 0,
            "buffer_radius_sqft": 1200,
            "qc_status": "NOT_VERIFIABLE",
            "bbox_or_mask": None,
            "image_metadata": {"source": "ESRI", "capture_date": datetime.now().strftime("%Y-%m-%d")},
            "tile_image": tile_path.replace("\\", "/"),
            "overlay_image": None
        }

    # pick the first image (usually the overlay with boxes)
    saved_overlay_full = os.path.join(pred_dir, saved_images[0])

    # move/copy overlay to our static outputs as overlay.jpg
    try:
        os.replace(saved_overlay_full, overlay_path)
    except Exception:
        from shutil import copyfile
        copyfile(saved_overlay_full, overlay_path)

    # 3) If no boxes, return NOT_VERIFIABLE with overlay present
    if len(results[0].boxes) == 0:
        return {
            "sample_id": sample_id,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "has_solar": False,
            "confidence": 0,
            "pv_area_sqm_est": 0,
            "buffer_radius_sqft": 1200,
            "qc_status": "NOT_VERIFIABLE",
            "bbox_or_mask": None,
            "image_metadata": {"source": source_name, "capture_date": datetime.now().strftime("%Y-%m-%d")},
            "tile_image": tile_path.replace("\\", "/"),
            "overlay_image": overlay_path.replace("\\", "/")
        }

    # 4) Get top detection and compute area
    box = results[0].boxes[0]
    bbox = [float(x) for x in box.xyxy[0].tolist()]  # [x1,y1,x2,y2]
    conf = float(box.conf[0])
    area = bbox_area_sqm(bbox, lat, zoom=zoom)
    qc_status = "VERIFIABLE" if conf >= 0.35 else "NOT_VERIFIABLE"

    # Build final JSON exactly as you requested
    return {
        "sample_id": sample_id,
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "has_solar": True,
        "confidence": round(conf, 4),
        "pv_area_sqm_est": round(area, 3),
        "buffer_radius_sqft": 1200,
        "qc_status": qc_status,
        "bbox_or_mask": {"bbox": bbox},
        "image_metadata": {"source": source_name, "capture_date": datetime.now().strftime("%Y-%m-%d")},
        "tile_image": tile_path.replace("\\", "/"),
        "overlay_image": overlay_path.replace("\\", "/")
    }
