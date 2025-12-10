# backend/app.py
import os, sys, traceback
from flask import Flask, request, jsonify, render_template, send_from_directory
# Ensure backend package modules are importable
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# import pipeline runner
from pipeline.pipeline_esri import run_pipeline_single

app = Flask(__name__, template_folder="templates")

# Where pipeline saves images (pipeline_esri.py uses this path)
STATIC_OUTPUTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "static", "outputs"))
os.makedirs(STATIC_OUTPUTS_DIR, exist_ok=True)

# Serve tile & overlay images
@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(STATIC_OUTPUTS_DIR, filename)

# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Predict by lat/lon
@app.route("/predict", methods=["POST"])
def predict():
    try:
        lat = float(request.form.get("latitude"))
        lon = float(request.form.get("longitude"))

        # call pipeline (sample_id = 1 for single queries; you can change logic)
        result = run_pipeline_single(sample_id=1, lat=lat, lon=lon)

        # Ensure overlay/tile paths are served as URLs
        # pipeline returns absolute paths; convert to relative URLs for browser
        if "tile_image" in result and result["tile_image"]:
            # tile_image -> '/outputs/tile.jpg'
            tile_name = os.path.basename(result["tile_image"])
            result["tile_image_url"] = f"/outputs/{tile_name}"
        else:
            result["tile_image_url"] = None

        if "overlay_image" in result and result["overlay_image"]:
            overlay_name = os.path.basename(result["overlay_image"])
            result["overlay_image_url"] = f"/outputs/{overlay_name}"
        else:
            result["overlay_image_url"] = None

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
