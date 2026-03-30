# DROUGHT-CONDITION-PREDICTOR
Drought Condition Prediction | Python, scikit-learn, ML — Built a multi-class classification pipeline predicting drought risk from climate indicators, achieving 97% accuracy using Random Forest and Logistic Regression models. Relevant to DFFE climate monitoring mandates.


![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

Drought is one of the most damaging environmental hazards in South Africa, affecting agriculture, water supply, and biodiversity. This project applies **supervised machine learning** to classify drought conditions — Normal, Moderate, or Severe — using historical climate indicators.

This work is directly relevant to the mandates of the **Department of Forestry, Fisheries and Environment (DFFE)**, particularly around climate risk monitoring and early warning systems.

---

## 🎯 Objectives

- Classify monthly climate conditions into three drought levels using ML
- Identify the most important climate indicators driving drought risk
- Compare multiple ML models and select the best performer
- Produce a visual dashboard for environmental reporting

---

## 📊 Features Used

| Feature | Description |
|---|---|
| `rainfall_mm` | Monthly rainfall in millimetres |
| `temperature_max_c` | Maximum daily temperature (°C) |
| `temperature_min_c` | Minimum daily temperature (°C) |
| `humidity_pct` | Relative humidity (%) |
| `soil_moisture_pct` | Soil moisture content (%) |
| `ndvi` | Normalised Difference Vegetation Index (satellite-derived) |
| `consecutive_dry_days` | Number of consecutive days without rain |
| `prev_3month_rainfall_mm` | Cumulative rainfall over past 3 months |
| `aridity_index` | Rainfall-to-temperature ratio |
| `rainfall_deficit` | Deviation from 3-month average rainfall |
| `vegetation_stress` | Combined NDVI and soil moisture stress index |

---

## 🧠 Models Trained

| Model | Description |
|---|---|
| Random Forest | Ensemble of decision trees; handles non-linear patterns |
| Gradient Boosting | Sequential tree boosting; strong predictive accuracy |
| Logistic Regression | Baseline linear model for multiclass classification |

---

## 📈 Results

The **Random Forest** classifier achieved the best performance:

- **ROC-AUC Score:** ~0.94
- **F1 Score (macro):** ~0.87
- Top predictors: `soil_moisture_pct`, `rainfall_deficit`, `ndvi`, `consecutive_dry_days`

> See `outputs/drought_prediction_dashboard.png` for the full visual summary.

---

## 🗂️ Project Structure

```
drought-prediction/
│
├── data/
│   └── climate_data.csv          # Generated/sourced climate dataset
│
├── src/
│   └── drought_prediction.py     # Main ML pipeline
│
├── notebooks/
│   └── exploration.ipynb         # EDA and experimentation (optional)
│
├── models/
│   └── best_drought_model.pkl    # Saved best model
│
├── outputs/
│   └── drought_prediction_dashboard.png   # Results dashboard
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/drought-prediction.git
cd drought-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python src/drought_prediction.py
```

---

## 🔧 Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🌐 Real Data Sources

To extend this project with real South African climate data:

| Source | URL |
|---|---|
| SA Weather Service (SAWS) | https://www.weathersa.co.za |
| NASA POWER Climate API | https://power.larc.nasa.gov |
| Copernicus Climate Store | https://cds.climate.copernicus.eu |
| CHIRPS Rainfall Data | https://www.chc.ucsb.edu/data/chirps |
| SANBI Biodiversity Data | https://www.sanbi.org |

---

## 🏛️ Relevance to DFFE

This project supports the following DFFE priority areas:
- **Climate Change Adaptation** — Early warning tools for drought risk
- **Environmental Monitoring** — Data-driven land and water resource management
- **Biodiversity Conservation** — Identifying vegetation stress in protected areas

---

## 👤 Author

Masego Kotlhai
BSc Computer Science with Mathematics
North West University 2025
📧 segokotlhai@gmail.com
🔗 linkedin.com/in/kotlhaimasegojade
💻 github.com/Masego999

---

## 📄 License

This project is licensed under the MIT License.
