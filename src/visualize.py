import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from folium.plugins import HeatMap, MarkerCluster
from sklearn.metrics import roc_curve


RISK_COLORS = {"Low": "#1a9850", "Medium": "#fee08b", "High": "#d73027"}


def plot_dispute_map(gdf, output_path):
    center = [gdf["latitude"].mean(), gdf["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron")

    heat_data = [
        [row["latitude"], row["longitude"], row["dispute_probability"]]
        for _, row in gdf[gdf["dispute_probability"] > 0.3].iterrows()
    ]
    if heat_data:
        HeatMap(heat_data, radius=14, blur=10, name="Dispute Risk Density").add_to(m)

    cluster = MarkerCluster(name="Land Parcels").add_to(m)
    for _, row in gdf.iterrows():
        risk = str(row.get("dispute_risk_level", "Low"))
        color = RISK_COLORS.get(risk, "#1a9850")
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(4, min(10, row["area_sqm"] / 5000)),
            color=color,
            fill=True,
            fill_opacity=0.75,
            popup=folium.Popup(
                f"<b>Parcel:</b> {row['parcel_id']}<br>"
                f"<b>State:</b> {row['state']}<br>"
                f"<b>Risk Level:</b> {risk}<br>"
                f"<b>Dispute Prob:</b> {row['dispute_probability']:.2%}<br>"
                f"<b>Land Use:</b> {row['land_use']}<br>"
                f"<b>Owners:</b> {row['n_registered_owners']}<br>"
                f"<b>Area:</b> {row['area_sqm']:,.0f} m²<br>"
                f"<b>Boundary Overlap:</b> {row['boundary_overlap_ratio']:.2%}",
                max_width=260
            )
        ).add_to(cluster)

    folium.LayerControl().add_to(m)
    m.save(output_path)
    print(f"Land dispute map saved: {output_path}")


def plot_overlap_analysis(df, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    risk_counts = df["dispute_risk_level"].value_counts()
    colors = [RISK_COLORS.get(str(k), "#ccc") for k in risk_counts.index]
    axes[0, 0].pie(risk_counts.values, labels=risk_counts.index,
                   colors=colors, autopct="%1.1f%%", startangle=90)
    axes[0, 0].set_title("Dispute Risk Level Distribution", fontsize=13, fontweight="bold")

    state_risk = df.groupby("state")["dispute_probability"].mean().sort_values(ascending=False)
    state_risk.plot(kind="barh", color="#d73027", ax=axes[0, 1])
    axes[0, 1].set_title("Avg Dispute Risk by State", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Avg Dispute Probability")

    land_use_risk = df.groupby("land_use")["dispute_probability"].mean().sort_values(ascending=False)
    land_use_risk.plot(kind="bar", color="#2166ac", ax=axes[1, 0], rot=30)
    axes[1, 0].set_title("Dispute Risk by Land Use Type", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("Avg Dispute Probability")

    axes[1, 1].scatter(df["boundary_overlap_ratio"], df["dispute_probability"],
                       c=df["n_registered_owners"], cmap="YlOrRd", alpha=0.4, s=8)
    axes[1, 1].set_xlabel("Boundary Overlap Ratio")
    axes[1, 1].set_ylabel("Dispute Probability")
    axes[1, 1].set_title("Overlap vs Dispute Risk (color=owners)", fontsize=13, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap="YlOrRd")
    sm.set_array(df["n_registered_owners"])
    plt.colorbar(sm, ax=axes[1, 1], label="Number of Owners")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Overlap analysis chart saved: {output_path}")


def plot_feature_importance(fi_df, output_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=fi_df, x="importance", y="feature", palette="RdYlGn_r", ax=ax)
    ax.set_title("Feature Importance — Land Dispute Model", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curve(y_test, y_prob, output_path):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#d73027", lw=2, label="ROC Curve")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Land Dispute Prediction")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
