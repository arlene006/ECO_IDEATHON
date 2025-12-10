# backend/fetch_tile_esri.py
import math, requests, os
from PIL import Image
from io import BytesIO

def _meters_per_pixel(lat, zoom):
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)

def fetch_esri_tile(lat, lon, out_path, zoom=19, size=1024, timeout=15):
    """
    Fetch ESRI World Imagery tile using the 'export' endpoint, using a bbox computed
    so the returned image centers on (lat,lon).
    size: number of pixels (square). ESRI may restrict the maximum; 1024 normally OK.
    """
    # compute half-width in meters for half the image: (size/2) * meters_per_pixel
    mpp = _meters_per_pixel(lat, zoom)
    half_width_m = (size / 2.0) * mpp

    # convert meters -> degrees (approx)
    deg_lat = half_width_m / 110574.0
    deg_lon = half_width_m / (111320.0 * math.cos(math.radians(lat)))

    bbox = f"{lon - deg_lon},{lat - deg_lat},{lon + deg_lon},{lat + deg_lat}"

    url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    params = {
        "bbox": bbox,
        "bboxSR": 4326,
        "size": f"{size},{size}",
        "imageSR": 4326,
        "format": "jpg",
        "f": "image",
        "dpi": 96
    }

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()

    img = Image.open(BytesIO(r.content)).convert("RGB")
    img.save(out_path)
    return out_path
