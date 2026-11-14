# Pharmacy Medicine Demand & Expiry-Risk Forecasting

An end-to-end Machine Learning system for retail pharmacies providing **daily medicine demand forecasting** and **batch-level expiry-risk classification**.

> **Note on Data Provenance**: The dataset used in this project is **synthetic** and designed for educational and demonstration purposes. It simulates sales transactions, purchase receipts, and monthly batch inventory counts across 2 pharmacy branches and 15 pharmaceutical products over a 2-year horizon (2024–2025).

---

## 1. Problem Statement

Pharmacy inventory management faces two conflicting challenges:
1. **Understocking (Stockouts)**: Depleting life-saving or chronic-care medicines compromises patient health and results in lost retail revenue.
2. **Overstocking (Expiry Loss)**: Holding excess batches with limited shelf life leads to product expiration, mandatory destruction costs, and regulatory compliance hazards.

This project implements two complementary ML models:
1. **Daily Demand Forecaster**: Predicts unit sales per store and medicine to inform optimal replenishment.
2. **Batch Expiry-Risk Classifier**: Identifies at snapshot time which active batches will deplete dangerously close to their expiration date ($\le 60$ days buffer) or expire with unsold stock under First-Expiry-First-Out (FEFO) dispensing.

---

## 2. Project Architecture & Structure

The codebase is deliberately organized to be clean, simple, and maintainable, avoiding unnecessary abstractions:

```text
Pharmacy_Forecasting/
│
├── data/
│   ├── raw/
│   │   ├── inventory_snapshots.csv     # 24 monthly operational batch stock counts
│   │   ├── products.csv                # Product catalog (shelf life, base demand, price)
│   │   ├── purchase_receipts.csv       # Purchase orders with manufacture & expiry dates
│   │   ├── sales_transactions.csv      # 42,329 transaction-level line items
│   │   └── stores.csv                  # Store master (ST_01 Central, ST_02 Community)
│   └── processed/
│       ├── daily_demand_features.csv   # Leakage-free daily store-product feature matrix
│       └── batch_expiry_features.csv   # Batch-level snapshot features & FEFO ground truth
│
├── notebooks/
│   ├── 01_demand_forecasting.ipynb     # Time-series demand forecasting & tuning
│   └── 02_expiry_risk_prediction.ipynb # FEFO simulation, group-aware CV & classification
│
├── models/
│   ├── demand_forecast_model.joblib    # Serialized champion demand regressor
│   ├── expiry_risk_model.joblib        # Serialized champion expiry classifier
│   ├── demand_forecast_diagnostics.png # Holdout residuals & time-series plots
│   └── expiry_risk_diagnostics.png     # PR curve, ROC curve, and confusion matrix
│
├── app.py                              # Streamlit decision support web application
├── requirements.txt                    # Core Python dependencies
└── README.md                           # Project documentation
```

---

## 3. Data Engineering & Integrity Assurance

### Full Cartesian Grid & Zero Imputation
Raw transaction logs only capture days with non-zero sales (21,500 series observations). Out of the expected 21,930 observations ($2\text{ stores} \times 15\text{ products} \times 731\text{ days}$), **430 store-product-day combinations had zero sales (1.96%)**.
- Omitting zero days corrupts lag and rolling window calculations.
- We reindexed the data to the full Cartesian grid with `quantity_sold = 0` before computing temporal features.

### FEFO Depletion Verification
A day-by-day First-Expiry-First-Out (FEFO) depletion algorithm was executed across all 709 purchase receipts and 42,329 sales transactions. Checking against all 1,092 month-end inventory snapshot counts yielded **0 mismatches**, proving the exactness of the reconstructed inventory mechanics.

### Handling Right-Censoring
Out of 1,092 inventory snapshot records:
- **660 records** represent batches that expire on or before 2025-12-31 (observable uncensored ground truth).
- **432 records** represent batches expiring after 2025-12-31. Because our sales window ends on 2025-12-31, their final outcome cannot be observed.
- Treating unobserved batches as "safe" would inject false negative labels. We explicitly isolate and exclude right-censored batches from supervised classifier training.

---

## 4. Feature Engineering

All features are engineered strictly from past information to eliminate lookahead leakage:

### Demand Forecasting Features
- **Lags**: `lag_1` (yesterday's momentum), `lag_7` (weekly seasonality), `lag_14`, `lag_28`.
- **Rolling Statistics**: Mean and standard deviation over 7, 14, and 28 days computed strictly on the `shift(1)` series (preventing today's target from leaking into its own rolling summary).
- **Calendar & Cyclical Signals**: `day_of_week`, `day_of_month`, `month`, `is_weekend`, plus sine/cosine cyclical transforms (`sin_dow`, `cos_dow`, `sin_month`, `cos_month`).
- **Catalog Attributes**: `base_daily_demand`, `shelf_life_days`, `unit_price`, `unit_cost`.

### Expiry-Risk Features
- **Temporal & Batch State**: `days_to_expiry`, `batch_age`, `quantity_on_hand`, `remaining_shelf_life_ratio` ($\frac{\text{days\_to\_expiry}}{\text{days\_to\_expiry} + \text{batch\_age}}$).
- **Demand Velocity**: `hist_avg_demand_30d` and `hist_std_demand_30d` computed strictly on past sales preceding `snapshot_date`.
- **Stock Coverage & Pressure**:
  $$\text{est\_demand\_before\_expiry} = \text{hist\_avg\_demand\_30d} \times \text{days\_to\_expiry}$$
  $$\text{coverage\_ratio} = \frac{\text{quantity\_on\_hand}}{\text{est\_demand\_before\_expiry} + \epsilon}$$

---

## 5. Validation Strategies

### Problem 1: Demand Forecasting
- **Split Strategy**: Strictly chronological.
  - **Train**: 2024-01-29 to 2025-06-30 (15,570 samples)
  - **Validation**: 2025-07-01 to 2025-09-30 (Q3 2025, 2,760 samples)
  - **Final Holdout Test**: 2025-10-01 to 2025-12-31 (Q4 2025, 2,760 samples, untouched until final evaluation)
- **Cross-Validation**: 3-fold expanding-window `TimeSeriesSplit` on the training set.

### Problem 2: Expiry-Risk Prediction
- **Split Strategy**: Chronological batch-exclusive holdout.
  - **Train**: 2024 snapshots (458 samples, 330 unique batches, 4.6% positive rate)
  - **Final Holdout Test**: 2025 brand-new batches (167 samples, 85 unique batches, 18.0% positive rate)
  - **Batch Overlap**: Exactly **0** batches shared between Train and Test.
- **Cross-Validation**: 5-fold `StratifiedGroupKFold` grouped by `batch_id` to prevent identical batches sighted across multiple monthly snapshots from leaking between folds.

---

## 6. Evaluation Results

### Demand Forecasting (Test Holdout: Q4 2025)

$$\text{WAPE} = \frac{\sum |y - \hat{y}|}{\sum y}$$

| Model Family | Validation MAE | Validation RMSE | Validation WAPE | Test Holdout MAE | Test Holdout WAPE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (Lag 7) | 4.542 | 5.952 | 30.26% | — | — |
| Baseline (7d Rolling Mean) | 3.661 | 4.772 | 24.39% | — | — |
| Ridge Regression (L2) | 3.396 | 4.385 | 22.63% | — | — |
| Random Forest Regressor | 3.291 | 4.231 | 21.93% | — | — |
| **HistGradientBoosting (Tuned)** | **3.259** | **4.216** | **21.71%** | **3.119 units** | **20.20%** |

*Explainability*: Permutation importance confirms `rolling_mean_28` (+1.19 MAE degradation), `lag_1` (+0.67 MAE), and `rolling_mean_7` (+0.53 MAE) are the strongest predictors, with `cos_dow` capturing the weekend shopping peak.

---

### Expiry-Risk Classification (Test Holdout: 2025 Batches)

Class weighting set to `balanced` to handle minority positive rates.

| Classifier | Recall (At Risk) | Precision (At Risk) | F1-Score | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Tuned)** | **1.000** | **0.566** | **0.723** | **0.931** | **0.980** |
| Random Forest (Balanced) | 0.733 | 0.550 | 0.629 | 0.716 | 0.925 |
| HistGradientBoosting (Balanced) | 0.733 | 0.688 | 0.710 | 0.653 | 0.931 |

#### Confusion Matrix (Tuned Logistic Regression on Test Holdout)

```text
                  Predicted Safe    Predicted At Risk
True Safe (137)        114                 23
True At Risk (30)        0                 30
```
- **False Negatives = 0**: The model successfully flagged 100% (30 out of 30) of at-risk batches.
- In medicine safety, eliminating false negatives is the priority: dispensing expired medicines has severe health and legal repercussions.

#### Key Risk Drivers (Odds Ratios)
- **`coverage_ratio` (Odds Ratio = 4.14x)**: Every 1 standard deviation increase in batch stock relative to expected sales before expiry quadruples the odds of near-expiry risk.
- **`store_id_ST_02` (Odds Ratio = 2.11x)**: Community Pharmacy ST_02 has lower sales velocity, doubling risk.
- **`remaining_shelf_life_ratio` (Odds Ratio = 0.38x)**: Higher remaining shelf life reduces expiry odds by 62%.

---

## 7. Streamlit Web Application

The project includes an interactive web dashboard in `app.py`:
- **Demand Forecasting**: Explore store/medicine sales trends, view actual vs. predicted values, and run custom single-day demand inferences.
- **Batch Inventory Health**: Review active inventory by store, inspect real-time ML risk scores (% probability), and test hypothetical scenarios using the interactive batch simulator.
- **Validation Analytics**: Inspect holdout test metrics, confusion matrices, and explainability plots directly within the UI.

To launch the dashboard:
```bash
streamlit run app.py
```

---

## 8. Instructions to Run

### 1. Environment Setup
```bash
git clone <repo-url>
cd Pharmacy_Forecasting
pip install -r requirements.txt
```

### 2. Execute Jupyter Notebooks
Run the notebooks in sequence:
```bash
# 1. Demand forecasting, feature engineering, and model training
jupyter notebook notebooks/01_demand_forecasting.ipynb

# 2. FEFO depletion target construction, group-aware CV, and classification
jupyter notebook notebooks/02_expiry_risk_prediction.ipynb
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```

---

## 9. Limitations & Practical Considerations

1. **Synthetic Nature**: The data generation assumes consistent seasonal cycles and deterministic FEFO batch depletion. In real-world retail pharmacies, dispensing staff may occasionally pick accessible batches rather than strict FEFO.
2. **Right-Censoring**: In production, continuous monitoring pipelines should incorporate survival analysis (e.g., Cox Proportional Hazards) to model time-to-depletion on currently active batches.
3. **Decision-Support Scope**: The expiry model is an operational tool for inventory reallocation, discounting, and supplier returns; it is not automated clinical advice.
