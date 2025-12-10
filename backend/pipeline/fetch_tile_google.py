# backend/fetch_tile_google.py
import math, os, requests
from PIL import Image
from io import BytesIO

def _deg2num(lat_deg, lon_deg, zoom):
    """Convert lat/lon to tile x,y at zoom (Google tile numbers)"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def fetch_google_tile(lat, lon, out_path, zoom=19, size=1024, timeout=15):
    """
    Fetch Google map tiles and compose a square image centered on lat/lon. This uses public
    tile endpoints (no API key). Works as a fallback in many environments.
    """
    # tile size from Google servers is 256 px per tile
    tile_px = 256
    # how many tiles needed to create a ~size x size image
    tiles_needed = max(1, int(math.ceil(size / tile_px)))
    if tiles_needed % 2 == 0:
        tiles_needed += 1  # keep odd so we can center

    center_x, center_y = _deg2num(lat, lon, zoom)

    # build a grid of tiles
    half = tiles_needed // 2
    from PIL import Image
    canvas_size = tile_px * tiles_needed
    canvas = Image.new("RGB", (canvas_size, canvas_size))

    for dx in range(-half, half+1):
        for dy in range(-half, half+1):
            x = center_x + dx
            y = center_y + dy
            tile_url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}"
            r = requests.get(tile_url, timeout=timeout)
            r.raise_for_status()
            tile = Image.open(BytesIO(r.content)).convert("RGB")
            canvas.paste(tile, ((dx+half)*tile_px, (dy+half)*tile_px))

    # Now crop center to requested size
    cx = canvas_size // 2
    cy = canvas_size // 2
    half_px = size // 2
    cropped = canvas.crop((cx - half_px, cy - half_px, cx + half_px, cy + half_px))
    cropped.save(out_path)
    return out_path
