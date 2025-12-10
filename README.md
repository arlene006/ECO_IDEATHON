Solar Rooftop Detection – Eco Ideathon 2025 Submission

This repository contains the full pipeline, model, environment setup, and prediction artefacts required for the Solar Rooftop Detection Challenge.
The system uses YOLOv8 + ESRI Satellite Tiles + a custom inference pipeline to detect rooftop solar panels, estimate area, and produce the mandated JSON output format.

📁 Repository Structure
root/
├── backend/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── fetch_tile_esri.py
│   │   ├── fetch_tile_google.py
│   │   ├── pipeline_esri.py
│   │   ├── train_yolo.py
│   │   └── export_training_logs.py
│   │
│   ├── batch_outputs/
│   │   ├── overlays/
│   │   ├── tiles/
│   │   └── predictions.json
│   │
│   ├── runs/detect/solar_detector/
│   │   └── weights/
│   │       └── best.pt      ← (Tracked with Git LFS)
│   │
│   ├── static/outputs/
│   ├── templates/
│   │   └── index.html
│   │
│   ├── training_logs/
│   │   ├── training_metrics.csv
│   │   ├── loss_curve.png
│   │   ├── f1_curve.png
│   │   └── rmse_per_epoch.csv
│   │
│   ├── app.py
│   └── batch_predict.py
│
├── environment_details/
│   ├── requirements.txt
│   ├── environment.yml
│   └── python_version.txt
│
├── model_card/
│   └── model_card.pdf
│
└── README.md


🚀 1. Setup Instructions
A. Using pip (Recommended)
pip install -r environment_details/requirements.txt

B. Using Conda
conda env create -f environment_details/environment.yml
conda activate solar-detection

⚙️ 2. How to Run the Web App
python pipeline_code/app.py


This will start the Flask server at:

👉 http://127.0.0.1:5000

The interface supports:

Search by Latitude & Longitude
Automatic Tile Fetching (ESRI 1024×1024)
YOLO-based detection
Overlay image + raw tile image
Full JSON output

### Batch Inference Output (Submission File)

The final predictions for all samples (3000 rows) are stored in:

backend/batch_outputs/predictions.json

This file contains:
- has_solar prediction
- pv_area_sqm_est
- confidence score
- bounding box (if present)
- qc_status
- tile and overlay image paths
- metadata
