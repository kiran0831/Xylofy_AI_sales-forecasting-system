# Sales Forecasting & Demand Intelligence System

An end-to-end sales forecasting and demand intelligence system built on 4 years of Superstore sales data (2015–2018). Combines time series forecasting, anomaly detection, and product clustering into a deployed interactive dashboard.

## What it does
- **Time series analysis** — trend/seasonality decomposition, stationarity testing (ADF)
- **3-model forecasting comparison** — SARIMA, Facebook Prophet, XGBoost (lag features), evaluated on MAE/RMSE/MAPE
- **Anomaly detection** — Isolation Forest + rolling Z-score on weekly sales
- **Product demand segmentation** — K-Means clustering (4 segments) with PCA visualization
- **Interactive dashboard** — 4-page Streamlit app (Sales Overview, Forecast Explorer, Anomaly Report, Product Demand Segments)

## Results
- **Best forecasting model:** SARIMA (17.7% MAPE), narrowly ahead of XGBoost (18.0%) and Prophet (21.9%)
- **Strongest growth segments:** East region, Technology category
- **4 product demand clusters:** High Volume Stable, Growing High-Value, Low Volume Steady, Declining

## Tech stack
Python · Pandas · Statsmodels · pmdarima · Prophet · XGBoost · Scikit-learn · Streamlit · Plotly

## Files
- `analysis.ipynb` — full analysis notebook (all tasks, with markdown explanations)
- `app.py` — Streamlit dashboard
- `train.csv`, `vgsales.csv` — datasets used
- `requirements.txt` — dependencies
- `summary.pdf` — 2-page executive business report
- `charts/` — exported chart images

## Run locally
\`\`\`
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Live demo
https://xylofyaisales-forecasting-system.streamlit.app
