# Diwali Air Quality Insights Dashboard

This Streamlit app explores air quality data around Diwali using PM2.5, PM10, temperature, and humidity from `feeds.csv`.

## Files
- `dashboard.py` — main Streamlit application
- `streamlit_app.py` — convenience entrypoint; Streamlit Cloud detects this by default
- `feeds.csv` — input data (must be present alongside the app)
- `requirements.txt` — dependencies for deployment

## Run locally

Prereqs: Python 3.11+ recommended.

```bash
# optional: create a fresh venv
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
streamlit run dashboard.py  # or: streamlit run streamlit_app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).

## Deploy (Streamlit Community Cloud)
1. Push this folder to a public GitHub repo.
2. In Streamlit Cloud, create a new app pointing to the repo.
3. App file: `streamlit_app.py` (or `dashboard.py` if you prefer)
4. The platform will install from `requirements.txt` automatically.

## Notes
- The app expects `feeds.csv` in the project root; place your latest export there.
- Smoothing is a time-based rolling mean (hours) computed on a datetime index.
- AQI categories follow CPCB thresholds for PM2.5.
