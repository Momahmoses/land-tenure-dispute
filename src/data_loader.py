import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


STATES = [
    "Lagos", "Abuja", "Rivers", "Ogun", "Delta", "Enugu",
    "Oyo", "Kano", "Kaduna", "Anambra", "Imo", "Cross River"
]
LAND_USE_TYPES = ["residential", "commercial", "agricultural", "industrial", "forest", "mixed"]


def generate_parcel_data(n_parcels=3000, seed=42):
    np.random.seed(seed)
    data = {
        "parcel_id": [f"PCL_{i:06d}" for i in range(n_parcels)],
        "state": np.random.choice(STATES, n_parcels),
        "latitude": np.random.uniform(4.5, 12.0, n_parcels),
        "longitude": np.random.uniform(3.0, 9.5, n_parcels),
        "area_sqm": np.random.lognormal(mean=7.5, sigma=1.5, size=n_parcels),
        "land_use": np.random.choice(LAND_USE_TYPES, n_parcels,
                                      p=[0.35, 0.20, 0.25, 0.08, 0.07, 0.05]),
        "n_registered_owners": np.random.poisson(lam=1.4, size=n_parcels).clip(1, 8),
        "record_age_years": np.random.exponential(scale=15, size=n_parcels).clip(0, 80),
        "boundary_surveys": np.random.poisson(lam=1.2, size=n_parcels).clip(0, 6),
        "inheritance_transfer": np.random.binomial(1, 0.35, n_parcels),
        "govt_acquisition_flag": np.random.binomial(1, 0.05, n_parcels),
        "urban_distance_km": np.random.exponential(scale=25, size=n_parcels),
        "boundary_overlap_ratio": np.random.beta(1.5, 8, n_parcels),
        "ownership_gap_years": np.random.exponential(scale=8, size=n_parcels).clip(0, 40),
        "prior_dispute_flag": np.random.binomial(1, 0.15, n_parcels),
        "documentation_score": np.random.beta(3, 2, n_parcels),
        "satellite_land_use_match": np.random.binomial(1, 0.75, n_parcels),
    }
    df = pd.DataFrame(data)
    df["land_use_code"] = pd.Categorical(df["land_use"]).codes

    dispute_score = (
        0.30 * df["boundary_overlap_ratio"]
        + 0.20 * (df["n_registered_owners"] - 1) / 7
        + 0.15 * df["prior_dispute_flag"]
        + 0.10 * df["inheritance_transfer"]
        + 0.10 * (1 - df["documentation_score"])
        + 0.08 * df["ownership_gap_years"] / 40
        + 0.05 * df["govt_acquisition_flag"]
        + 0.02 * (df["record_age_years"] / 80)
        + np.random.normal(0, 0.04, n_parcels)
    ).clip(0, 1)

    df["dispute_score"] = dispute_score
    df["disputed"] = (dispute_score > 0.5).astype(int)
    return df


def classify_dispute_risk(df, config):
    df = df.copy()
    t = config["thresholds"]
    df["dispute_risk_level"] = pd.cut(
        df["dispute_score"],
        bins=[-np.inf, t["medium_dispute_risk"], t["high_dispute_risk"], np.inf],
        labels=["Low", "Medium", "High"]
    )
    return df


def to_geodataframe(df):
    return gpd.GeoDataFrame(
        df,
        geometry=[Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326"
    )
