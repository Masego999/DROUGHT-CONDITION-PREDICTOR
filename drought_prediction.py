"""
Drought Condition Prediction Using Historical Climate Data
Author:     Masego Kotlhai
- Portfolio Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score
)
from sklearn.pipeline import Pipeline
import joblib
import warnings
import os
warnings.filterwarnings("ignore")

# Base directory (works regardless of where script is run from)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── 1. GENERATE REALISTIC SOUTH AFRICAN CLIMATE DATA ─────────────────────────
# In a real project, replace this with data from:
# - South African Weather Service (SAWS): https://www.weathersa.co.za
# - NASA POWER API: https://power.larc.nasa.gov
# - Copernicus Climate Data Store: https://cds.climate.copernicus.eu

np.random.seed(42)

def generate_sa_climate_data(n_samples: int = 1200) -> pd.DataFrame:
    """
    Simulate monthly climate data for South African regions.
    Features reflect real climatic indicators used in drought indices.
    """
    months = np.tile(np.arange(1, 13), n_samples // 12 + 1)[:n_samples]

    # Seasonal rainfall pattern (summer rainfall region)
    seasonal_rain = 60 + 40 * np.sin(2 * np.pi * (months - 1) / 12)

    data = pd.DataFrame({
        "month": months,
        "rainfall_mm": np.clip(
            seasonal_rain + np.random.normal(0, 25, n_samples), 0, None
        ),
        "temperature_max_c": 25 + 8 * np.sin(2 * np.pi * months / 12)
                              + np.random.normal(0, 2, n_samples),
        "temperature_min_c": 10 + 6 * np.sin(2 * np.pi * months / 12)
                              + np.random.normal(0, 1.5, n_samples),
        "humidity_pct": np.clip(
            55 - 15 * np.sin(2 * np.pi * months / 12)
            + np.random.normal(0, 8, n_samples), 10, 100
        ),
        "wind_speed_kmh": np.clip(
            np.random.normal(18, 6, n_samples), 0, None
        ),
        "soil_moisture_pct": np.clip(
            np.random.normal(35, 12, n_samples), 5, 80
        ),
        "ndvi": np.clip(                          # Vegetation health index
            0.4 + 0.2 * np.sin(2 * np.pi * (months - 2) / 12)
            + np.random.normal(0, 0.08, n_samples), 0, 1
        ),
        "consecutive_dry_days": np.clip(
            np.random.exponential(12, n_samples), 0, 90
        ).astype(int),
        "prev_3month_rainfall_mm": np.clip(
            seasonal_rain * 3 + np.random.normal(0, 40, n_samples), 0, None
        ),
    })

    # ── Derive drought label using SPI-like logic ──
    # Drought occurs when rainfall is low, soil is dry, and temperatures are high
    drought_score = (
        -0.4 * (data["rainfall_mm"] / data["rainfall_mm"].mean())
        -0.3 * (data["soil_moisture_pct"] / data["soil_moisture_pct"].mean())
        +0.2 * (data["temperature_max_c"] / data["temperature_max_c"].mean())
        +0.1 * (data["consecutive_dry_days"] / data["consecutive_dry_days"].mean())
    )

    # Three classes: 0 = Normal, 1 = Moderate Drought, 2 = Severe Drought
    data["drought_level"] = pd.cut(
        drought_score,
        bins=[-np.inf, -0.3, 0.2, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    return data


# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that improve model performance."""
    df = df.copy()

    # Temperature range (heat stress indicator)
    df["temp_range_c"] = df["temperature_max_c"] - df["temperature_min_c"]

    # Aridity index (simple ratio)
    df["aridity_index"] = df["rainfall_mm"] / (df["temperature_max_c"] + 0.001)

    # Rainfall deficit vs 3-month average
    df["rainfall_deficit"] = df["prev_3month_rainfall_mm"] / 3 - df["rainfall_mm"]

    # Vegetation stress (low NDVI + low soil moisture)
    df["vegetation_stress"] = (1 - df["ndvi"]) * (1 - df["soil_moisture_pct"] / 100)

    return df


# ── 3. TRAIN & EVALUATE MODELS ────────────────────────────────────────────────

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return the best one."""

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
    }

    scaler = StandardScaler()
    results = {}

    print("\n" + "="*55)
    print("   MODEL EVALUATION RESULTS")
    print("="*55)

    best_score = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        if name == "Logistic Regression":
            pipeline = Pipeline([("scaler", scaler), ("model", model)])
        else:
            pipeline = Pipeline([("model", model)])

        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_macro")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_score = roc_auc_score(y_test, pipeline.predict_proba(X_test), multi_class="ovr")

        results[name] = {
            "pipeline": pipeline,
            "y_pred": y_pred,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "roc_auc": test_score,
        }

        print(f"\n📊 {name}")
        print(f"   CV F1 Score : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"   ROC-AUC     : {test_score:.3f}")

        if test_score > best_score:
            best_score = test_score
            best_model = pipeline
            best_name = name

    print(f"\n✅ Best Model: {best_name} (ROC-AUC: {best_score:.3f})")
    print("="*55)

    return results, best_model, best_name


# ── 4. VISUALISATIONS ─────────────────────────────────────────────────────────

def plot_results(df, results, best_name, feature_names, output_dir="outputs"):
    """Generate and save all plots."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "Drought Condition Prediction — South Africa\nML Analysis Dashboard",
        fontsize=15, fontweight="bold", y=1.01
    )

    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    labels = ["Normal", "Moderate Drought", "Severe Drought"]

    # Plot 1: Drought class distribution
    ax = axes[0, 0]
    counts = df["drought_level"].value_counts().sort_index()
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_title("Drought Level Distribution", fontweight="bold")
    ax.set_ylabel("Number of Months")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha="center", fontsize=10)
    ax.set_ylim(0, counts.max() * 1.15)

    # Plot 2: Rainfall vs Soil Moisture (coloured by drought level)
    ax = axes[0, 1]
    for level, color, label in zip([0, 1, 2], colors, labels):
        mask = df["drought_level"] == level
        ax.scatter(df.loc[mask, "rainfall_mm"], df.loc[mask, "soil_moisture_pct"],
                   c=color, label=label, alpha=0.5, s=18)
    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("Soil Moisture (%)")
    ax.set_title("Rainfall vs Soil Moisture", fontweight="bold")
    ax.legend(fontsize=8)

    # Plot 3: Monthly average rainfall
    ax = axes[0, 2]
    monthly = df.groupby("month")["rainfall_mm"].mean()
    ax.bar(monthly.index, monthly.values, color="#3498db", edgecolor="white")
    ax.set_title("Average Monthly Rainfall", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Rainfall (mm)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=8)

    # Plot 4: Confusion matrix for best model
    ax = axes[1, 0]
    best_result = results[best_name]
    y_test_vals = best_result["y_pred"]  # reuse stored predictions
    cm = confusion_matrix(y_test_vals, y_test_vals)  # placeholder visual
    # Use actual confusion matrix stored in results
    cm = results[best_name].get("cm")
    if cm is not None:
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix\n({best_name})", fontweight="bold")
        ax.set_xticklabels(labels, rotation=15, fontsize=8)
        ax.set_yticklabels(labels, rotation=0, fontsize=8)

    # Plot 5: Feature importance (Random Forest)
    ax = axes[1, 1]
    rf_result = results.get("Random Forest")
    if rf_result:
        rf_model = rf_result["pipeline"].named_steps["model"]
        importances = pd.Series(rf_model.feature_importances_, index=feature_names)
        importances.nlargest(8).sort_values().plot(
            kind="barh", ax=ax, color="#9b59b6", edgecolor="white"
        )
        ax.set_title("Top Feature Importances\n(Random Forest)", fontweight="bold")
        ax.set_xlabel("Importance Score")

    # Plot 6: Model comparison
    ax = axes[1, 2]
    model_names = list(results.keys())
    roc_scores = [results[m]["roc_auc"] for m in model_names]
    bars = ax.bar(model_names, roc_scores,
                  color=["#3498db", "#e67e22", "#1abc9c"], edgecolor="white")
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Model Comparison (ROC-AUC)", fontweight="bold")
    ax.set_ylabel("ROC-AUC Score")
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, label="0.9 threshold")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, roc_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticklabels(model_names, rotation=10, fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drought_prediction_dashboard.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📈 Dashboard saved → {output_dir}/drought_prediction_dashboard.png")


# ── 5. MAIN PIPELINE ──────────────────────────────────────────────────────────

def main():
    print("\n🌍 Drought Condition Prediction — South Africa")
    print("   Portfolio Project | Dept. of Forestry, Fisheries & Environment\n")

    # Generate data
    print("📥 Loading climate data...")
    df = generate_sa_climate_data(n_samples=1200)
    df = engineer_features(df)
    df.to_csv(os.path.join(DATA_DIR, "climate_data.csv"), index=False)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Drought distribution:\n{df['drought_level'].value_counts().sort_index().to_string()}")

    # Prepare features
    feature_cols = [
        "rainfall_mm", "temperature_max_c", "temperature_min_c",
        "humidity_pct", "wind_speed_kmh", "soil_moisture_pct",
        "ndvi", "consecutive_dry_days", "prev_3month_rainfall_mm",
        "temp_range_c", "aridity_index", "rainfall_deficit", "vegetation_stress"
    ]

    X = df[feature_cols]
    y = df["drought_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n🔀 Train/Test Split: {len(X_train)} / {len(X_test)} samples")

    # Train models
    results, best_model, best_name = train_models(X_train, X_test, y_train, y_test)

    # Store confusion matrix for plotting
    for name, res in results.items():
        res["cm"] = confusion_matrix(y_test, res["y_pred"])

    # Detailed report for best model
    print(f"\n📋 Classification Report — {best_name}:")
    print(classification_report(
        y_test, results[best_name]["y_pred"],
        target_names=["Normal", "Moderate Drought", "Severe Drought"]
    ))

    # Save plots
    plot_results(df, results, best_name, feature_cols, OUTPUTS_DIR)

    # Save best model
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_drought_model.pkl"))
    print(f"💾 Model saved → {MODELS_DIR}/best_drought_model.pkl")

    print("\n✅ Pipeline complete! Check the outputs/ folder for your dashboard.\n")


if __name__ == "__main__":
    main()
