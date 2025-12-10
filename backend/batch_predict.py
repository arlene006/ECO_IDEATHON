import os
import math
import json
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from pipeline.fetch_tile_esri import fetch_esri_tile
from pipeline.fetch_tile_google import fetch_google_tile
from PIL import Image

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
MODEL_PATH = r"C:/Users/arlen/OneDrive/Desktop/eco_ideathon/backend/runs/detect/solar_detector/weights/best.pt"

OUTPUT_DIR = "batch_outputs"
TILE_SIZE = 1024
ZOOM = 19
CONF_THRESHOLD = 0.30

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "tiles"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "overlays"), exist_ok=True)

# Load YOLO model
MODEL = YOLO(MODEL_PATH)


# ---------------------------------------------------
# GEOSPATIAL HELPERS
# ---------------------------------------------------
def meters_per_pixel(lat, zoom):
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


def bbox_area_sqm(bbox, lat, zoom):
    x1, y1, x2, y2 = bbox
    px_area = max(1, (x2 - x1) * (y2 - y1))
    mpp = meters_per_pixel(lat, zoom)
    return px_area * (mpp ** 2)


# ---------------------------------------------------
# TILE FETCHER (ESRI → GOOGLE FALLBACK)
# ---------------------------------------------------
def fetch_tile(lat, lon, out_path):
    try:
        fetch_esri_tile(lat, lon, out_path, zoom=ZOOM, size=TILE_SIZE)
        return "ESRI"
    except Exception:
        try:
            fetch_google_tile(lat, lon, out_path, zoom=ZOOM, size=TILE_SIZE)
            return "GOOGLE_TILE"
        except:
            return None


# ---------------------------------------------------
# RUN YOLO INFERENCE
# ---------------------------------------------------
def run_yolo(tile_path, overlay_path, lat):
    results = MODEL.predict(
        source=tile_path,
        conf=CONF_THRESHOLD,
        project="runs/detect",
        name="batch_predict",
        exist_ok=True,
        save=True
    )

    pred_dir = results[0].save_dir
    saved_img = [f for f in os.listdir(pred_dir) if f.endswith(".jpg")][0]

    os.replace(os.path.join(pred_dir, saved_img), overlay_path)

    # NO detections
    if len(results[0].boxes) == 0:
        return None

    box = results[0].boxes[0]
    bbox = [float(x) for x in box.xyxy[0].tolist()]
    conf = float(box.conf[0])
    area = bbox_area_sqm(bbox, lat, ZOOM)

    return {
        "bbox": bbox,
        "conf": conf,
        "area": area
    }


# ---------------------------------------------------
# BUILD JSON STRUCTURE (MANDATORY FORMAT)
# ---------------------------------------------------
def build_record(sample_id, lat, lon, detection, source_name, tile_path, overlay_path):
    if detection is None:
        return {
            "sample_id": sample_id,
            "lat": lat,
            "lon": lon,
            "has_solar": False,
            "confidence": 0,
            "pv_area_sqm_est": 0,
            "buffer_radius_sqft": 1200,
            "qc_status": "NOT_VERIFIABLE",
            "bbox_or_mask": None,
            "image_metadata": {"source": source_name, "capture_date": None},
            "tile_image": tile_path.replace("\\", "/"),
            "overlay_image": overlay_path.replace("\\", "/")
        }

    return {
        "sample_id": sample_id,
        "lat": lat,
        "lon": lon,
        "has_solar": True,
        "confidence": detection["conf"],
        "pv_area_sqm_est": detection["area"],
        "buffer_radius_sqft": 1200,
        "qc_status": "VERIFIABLE" if detection["conf"] >= 0.35 else "NOT_VERIFIABLE",
        "bbox_or_mask": {"bbox": detection["bbox"]},
        "image_metadata": {"source": source_name, "capture_date": datetime.now().strftime("%Y-%m-%d")},
        "tile_image": tile_path.replace("\\", "/"),
        "overlay_image": overlay_path.replace("\\", "/")
    }


# ---------------------------------------------------
# MAIN BATCH PIPELINE
# ---------------------------------------------------
def run_batch(input_excel):
    df = pd.read_excel(input_excel)
    results = []

    for i, row in df.iterrows():
        # Accept multiple possible column names
        sample_id = (
        row.get("sample_id")
        or row.get("sampleid")
        or row.get("sampleId")
        or row.get("SampleID")
        or row.get("id")
        )

        # Accept multiple possible column names for latitude
        lat = float(
        row.get("lat")
        or row.get("latitude")
        or row.get("Latitude")
        or row.get("LAT")
        or row.get("Lat")
        )

# Accept multiple possible column names for longitude
        lon = float(
        row.get("lon")
        or row.get("longitude")
        or row.get("Longitude")
        or row.get("LON")
        or row.get("Lon")
    )


        print(f"[{i+1}/{len(df)}] Processing {sample_id}...")

        tile_path = os.path.join(OUTPUT_DIR, "tiles", f"{sample_id}.jpg")
        overlay_path = os.path.join(OUTPUT_DIR, "overlays", f"{sample_id}_overlay.jpg")

        source = fetch_tile(lat, lon, tile_path)

        if source is None:
            results.append(build_record(sample_id, lat, lon, None, None, tile_path, overlay_path))
            continue

        detection = run_yolo(tile_path, overlay_path, lat)

        record = build_record(sample_id, lat, lon, detection, source, tile_path, overlay_path)
        results.append(record)

    # Save final JSON
    json_path = os.path.join(OUTPUT_DIR, "predictions.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\nBatch Prediction Completed!")
    print("Saved:", json_path)


# ---------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------
if __name__ == "__main__":
    run_batch(r"C:/Users/arlen/OneDrive/Desktop/eco_ideathon/EI_train_data.xlsx")

