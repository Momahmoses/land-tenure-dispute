import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
import joblib


FEATURES = [
    "boundary_overlap_ratio", "n_registered_owners", "record_age_years",
    "boundary_surveys", "inheritance_transfer", "govt_acquisition_flag",
    "urban_distance_km", "ownership_gap_years", "prior_dispute_flag",
    "documentation_score", "satellite_land_use_match", "land_use_code",
    "area_sqm"
]
TARGET = "disputed"


def build_pipeline(config):
    clf = GradientBoostingClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        learning_rate=config["model"]["learning_rate"],
        random_state=config["model"]["random_state"]
    )
    return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])


def train(df, config):
    X = df[FEATURES].fillna(0)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y
    )
    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "cv_scores": cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring="roc_auc"
        ).tolist()
    }
    return pipeline, metrics, (X_test, y_test, y_prob)


def predict_dispute_risk(pipeline, df, config):
    df = df.copy()
    probs = pipeline.predict_proba(df[FEATURES].fillna(0))[:, 1]
    t = config["thresholds"]
    df["dispute_probability"] = probs
    df["dispute_risk_level"] = pd.cut(
        probs,
        bins=[-np.inf, t["medium_dispute_risk"], t["high_dispute_risk"], np.inf],
        labels=["Low", "Medium", "High"]
    )
    return df


def feature_importance(pipeline, top_n=13):
    clf = pipeline.named_steps["classifier"]
    return pd.DataFrame({
        "feature": FEATURES,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False).head(top_n)


def prioritize_resolution(df):
    high_risk = df[df["dispute_risk_level"] == "High"].copy()
    high_risk["resolution_urgency"] = (
        high_risk["dispute_probability"]
        * np.log1p(high_risk["area_sqm"])
        * (1 + high_risk["prior_dispute_flag"])
    )
    return high_risk.sort_values("resolution_urgency", ascending=False)


def save_model(pipeline, path):
    joblib.dump(pipeline, path)
