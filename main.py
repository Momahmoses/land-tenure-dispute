import os
from src.data_loader import load_config, generate_parcel_data, classify_dispute_risk, to_geodataframe
from src.model import train, predict_dispute_risk, feature_importance, prioritize_resolution, save_model
from src.visualize import plot_dispute_map, plot_overlap_analysis, plot_feature_importance, plot_roc_curve


def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    config = load_config("config.yaml")
    print(f"[1/5] Config loaded — Targets: {', '.join(config['targets'])}")

    df = generate_parcel_data(n_parcels=3000)
    df = classify_dispute_risk(df, config)
    disputed_pct = df["disputed"].mean()
    print(f"[2/5] {len(df):,} land parcels — {disputed_pct:.1%} flagged as disputed")

    pipeline, metrics, (X_test, y_test, y_prob) = train(df, config)
    print(f"[3/5] Model trained — AUC: {metrics['roc_auc']:.4f}")
    print(f"      CV AUC scores: {[round(s, 3) for s in metrics['cv_scores']]}")
    print(metrics["classification_report"])

    save_model(pipeline, config["output"]["model_path"])

    result_df = predict_dispute_risk(pipeline, df, config)
    result_df.to_csv(config["output"]["report"], index=False)
    print(f"[4/5] Predictions saved to {config['output']['report']}")

    priority_df = prioritize_resolution(result_df)
    print(f"      {len(priority_df)} high-risk parcels prioritized for resolution")
    print(f"      Top 5 urgent cases:")
    print(priority_df[["parcel_id", "state", "land_use", "dispute_probability",
                        "boundary_overlap_ratio", "n_registered_owners"]].head(5).to_string(index=False))

    fi_df = feature_importance(pipeline)
    plot_feature_importance(fi_df, "outputs/feature_importance.png")
    plot_roc_curve(y_test, y_prob, "outputs/roc_curve.png")
    plot_overlap_analysis(result_df, config["output"]["overlap_analysis"])

    gdf = to_geodataframe(result_df)
    plot_dispute_map(gdf, config["output"]["dispute_map"])
    print("[5/5] All outputs saved to /outputs/")
    print("\nDone.")


if __name__ == "__main__":
    main()
