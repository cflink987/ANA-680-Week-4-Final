# Gaming Addiction Risk API (Flask)

Predicts `high_addiction` (top 20% of addiction_level) from 12 deployment features using a trained scikit-learn Pipeline.

## Dataset
- Source file used in notebook: `gaming_mental_health_10M_40features.csv`
- Training/EDA used a 200k sample.
- Target: `high_addiction` = 1 if `addiction_level` >= 80th percentile threshold (printed as 4.44 in notebook)

## Deployment features (12)
age, gender, daily_gaming_hours, weekly_sessions, years_gaming, competitive_rank,
online_friends, microtransactions_spending, screen_time_total, sleep_hours, stress_level, depression_score

## Artifacts
- `artifacts/deploy_model.pkl` (joblib, sklearn Pipeline)
- `artifacts/deploy_features.json`
- `artifacts/high_addiction_threshold.txt`

## Run locally (Windows / PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py