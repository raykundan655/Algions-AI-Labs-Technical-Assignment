# Pharmacy Demand & Expiry-Risk Forecasting — Raw Synthetic Dataset

## Purpose
This dataset is designed for an end-to-end learning project with two tasks:
1. Daily medicine demand forecasting at store-product level.
2. Batch-level expiry-risk classification.

## Files
- `sales_transactions.csv`: transaction-line sales. Aggregate this by date, store and product for demand forecasting.
- `purchase_receipts.csv`: raw purchase receipt and batch details, including manufacture and expiry dates.
- `inventory_snapshots.csv`: month-end operational stock counts by batch. It contains raw inventory state, not an expiry-risk label.
- `products.csv`: product master data.
- `stores.csv`: store master data.

## What is intentionally NOT provided
- No precomputed lag or rolling-average features.
- No precomputed `days_to_expiry` feature.
- No `expiry_risk` target label.
- No train/test split and no model-ready feature matrix.

You are expected to construct these during the project.

## Synthetic provenance
All records are synthetically generated for educational use. They do not represent real pharmacies, real patients, or real commercial transactions. Random seed: 42. Sales were simulated with weekly/yearly patterns and a small promotion effect. Inventory sales were allocated to available batches using FEFO (first-expiry-first-out) during data generation. The batch allocation is not included in the sales file; reconstructing inventory outcomes is part of the exercise.

## Important modelling guidance
- Demand forecasting: aggregate transaction lines into daily units sold by `date`, `store_id`, and `product_id`. Create lag and rolling features using only past values (e.g. `shift(1)` before rolling).
- Use chronological train/validation/test splits. Do not randomly split time-series rows.
- Expiry risk: derive `days_to_expiry` from `expiry_date - snapshot_date`. Build historical demand features using sales strictly before `snapshot_date`.
- Define a transparent retrospective label using future sales between snapshot date and expiry date. A simple initial rule is to estimate how much of the batch can be sold before expiry and flag a batch if its snapshot quantity is greater than the realized sales available before expiry. However, because multiple batches of the same product can coexist, a stronger implementation should reconstruct FEFO batch depletion using the snapshot inventory, subsequent purchase receipts, and subsequent sales. Document the exact rule.
- Never use future sales or any information after the snapshot date as model input features. Future information may be used only to create retrospective training labels.
- A batch that expires after the dataset end may have an unobserved outcome; exclude such rows from supervised expiry-risk training or explicitly mark them as censored.
- The risk model is an inventory decision-support tool, not medical advice. Do not recommend dispensing expired medicines.

## Suggested project outputs
1. Data quality and EDA notebook.
2. Demand forecasting notebook and saved model.
3. Expiry-risk target construction and classification notebook.
4. Reproducible preprocessing/model pipeline, tests, API and dashboard.
