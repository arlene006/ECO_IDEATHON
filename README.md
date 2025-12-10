# Solar Rooftop Detection – Eco Ideathon 2025 Submission

This repository contains the full pipeline, model, environment setup, and prediction artefacts required for the Solar Rooftop Detection Challenge.  
The system uses YOLOv8 + ESRI Satellite Tiles + a custom inference pipeline to detect rooftop solar panels, estimate area, and produce the mandated JSON output format.

## 📁 Repository Structure

```text
root/
├── backend/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── pipeline_esri.py
│   │   ├── fetch_tile_esri.py
│   │   ├── fetch_tile_google.py
│   │   ├── train_yolo.py
│   │   └── export_training_logs.py
│   │
│   ├── batch_outputs/
│   │   ├── overlays/
│   │   ├── tiles/
│   │   └── predictions.json
│   │
│   ├── runs/
│   │   └── detect/
│   │       └── solar_detector/
│   │           └── weights/
│   │               └── best.pt
│   │
│   ├── static/
│   │   └── outputs/
│   │
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
```

---

## 🚀 Installation

### Using pip
```bash
pip install -r environment_details/requirements.txt
```

### Using Conda
```bash
conda env create -f environment_details/environment.yml
conda activate solar-detection
```

---

## 🖥 Run the Web App

```bash
python backend/app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

Features:
- Latitude/Longitude input
- ESRI tile fetching
- YOLOv8 detection
- Solar panel area estimation
- Tile + overlay image visualization
- Complete JSON output

---

## 📦 Batch Inference (3000 samples)

```bash
python backend/batch_predict.py
```

Output stored in:

```
backend/batch_outputs/predictions.json
```

Format includes:
- has_solar
- pv_area_sqm_est
- confidence
- bbox_or_mask
- qc_status
- tile & overlay paths

---

## 📊 Training Logs

Located in:

```
backend/training_logs/
```

Contains:
- training_metrics.csv  
- loss_curve.png  
- f1_curve.png  
- rmse_per_epoch.csv  

---

## 🧠 Model Card

```
model_card/model_card.pdf
```

---

## ✔ Deliverables Summary

| Deliverable | Included | Location |
|------------|----------|----------|
| Pipeline Code | ✔ | backend/pipeline |
| Training Logs | ✔ | backend/training_logs |
| Model File (YOLO) | ✔ | backend/runs/detect/solar_detector/weights/best.pt |
| Environment Files | ✔ | environment_details/ |
| Model Card | ✔ | model_card/model_card.pdf |
| Predictions JSON | ✔ | backend/batch_outputs/predictions.json |
| Web App | ✔ | backend/app.py |

---

