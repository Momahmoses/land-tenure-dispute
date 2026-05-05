# Land Tenure & Property Dispute Resolution System

A GIS + machine learning system that digitizes land parcel boundaries, flags overlapping or contested parcels, and prioritizes resolution cases — tackling Nigeria's land administration crisis with data-driven dispute detection.

## Features
- Synthetic land parcel dataset (3,000 parcels) with boundary, ownership, and documentation features
- Gradient Boosting dispute probability model
- Resolution urgency scoring (boundary overlap × area × prior disputes)
- Interactive dispute risk heatmap with per-parcel details
- State-level and land-use-level dispute analysis

## Project Structure
```
land-tenure-dispute/
├── src/
│   ├── data_loader.py    # Parcel data generation, dispute risk classification
│   ├── model.py          # GBM dispute predictor, resolution prioritization
│   └── visualize.py      # Dispute map, overlap analysis charts
├── data/raw/             # Land parcel records, ownership history, satellite classification
├── models/               # Saved model artifacts
├── outputs/              # Maps, reports, charts
├── config.yaml
├── main.py
└── requirements.txt
```

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Data Sources
| Layer | Source |
|-------|--------|
| Land parcels | Nigeria Land Registry / MLHUD |
| Boundary surveys | OSGOF (Office of the Surveyor General) |
| Satellite imagery | Sentinel-2 via Google Earth Engine |
| Ownership records | State Land Registries |

## Output
- `outputs/land_dispute_map.html` — interactive dispute risk map
- `outputs/land_dispute_report.csv` — per-parcel dispute probabilities
- `outputs/boundary_overlap_analysis.png` — overlap vs risk charts
- `outputs/feature_importance.png` — top dispute predictors
- `outputs/roc_curve.png` — model performance

## Author
MOMAH MOSES .C.
