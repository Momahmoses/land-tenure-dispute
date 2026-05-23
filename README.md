# Land Tenure & Property Dispute Resolution System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GIS + ML system that digitises land parcel boundaries, flags overlapping or contested parcels, and prioritises resolution cases, tackling Nigeria's land administration crisis with data-driven dispute detection and resolution scheduling.

---

## Problem Statement

Nigeria's land administration system is severely fragmented, enabling rampant land grabbing, double allocation, and boundary disputes. This system provides automated dispute detection and resolution prioritisation to support land registries and courts.

---

## Features

| Feature | Description |
|---------|-------------|
| 3,000 Parcel Synthetic Dataset | Boundary, ownership, documentation, and dispute features |
| Gradient Boosting Dispute Model | Probability score per parcel |
| Resolution Urgency Scoring | Boundary overlap × area × prior disputes |
| Interactive Dispute Risk Map | Folium heatmap with per-parcel details |
| State & Land-Use Analysis | Dispute rate breakdown by state and land-use type |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Machine Learning | scikit-learn (Gradient Boosting) |
| Geospatial | GeoPandas, Folium |
| Data | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |

---

## Project Structure

```
land-tenure-dispute/
├── src/
│   ├── data_loader.py     # Parcel data generation and dispute risk classification
│   ├── model.py           # GBM dispute predictor, resolution prioritisation
│   └── visualize.py       # Dispute map and overlap analysis charts
├── data/raw/              # Land parcel records, ownership history
├── models/                # Saved model artifacts
├── config.yaml            # Risk thresholds, state boundaries
├── main.py                # Pipeline entry point
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/land-tenure-dispute.git
cd land-tenure-dispute
pip install -r requirements.txt
python main.py
```

---

## Data Sources

- NLC Nigeria land parcel registry
- State land administration offices (Plateau, Benue, Kaduna, Niger)
- ACLED conflict event data (farmer-herder classification)
- GRID3 Nigeria administrative boundaries

---

## Author

**Momah Moses**, Geospatial AI Engineer & Data Scientist
[GitHub](https://github.com/Momahmoses) · [Portfolio](https://momahmoses-ng-gis-portfolio.hf.space)
