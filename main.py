import pandas as pd
import numpy as np
import joblib
import httpx
import os
import datetime
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import SessionLocal, engine
import models

# Buat tabel database
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForecastRequest(BaseModel):
    region_name: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

COORDINATES = {
    "Sumatra": {"lat": -0.5897, "lon": 101.3431},
    "Jawa": {"lat": -7.6145, "lon": 110.7122},
    "Kalimantan": {"lat": -0.2787, "lon": 111.4753},
    "Sulawesi": {"lat": -2.0833, "lon": 120.8333},
    "Nusa Tenggara": {"lat": -8.6500, "lon": 117.3667},
    "Maluku": {"lat": -3.2333, "lon": 130.1500},
    "Papua": {"lat": -4.2667, "lon": 138.0833},
}

REGION_TIME_OFFSETS = {"Sumatra": 7, "Jawa": 7, "Kalimantan": 8, "Sulawesi": 8, "Nusa Tenggara": 8, "Maluku": 9, "Papua": 9}
TARGET_COL = "ALLSKY_SFC_SW_DWN"
BASE_FEATURES = ["CLOUD_AMT", "T2M", "RH2M", "PS", "CLRSKY_SFC_SW_DWN"]
LOOKBACK = 24

def create_time_features(df: pd.DataFrame, region_name: str) -> pd.DataFrame:
    out = df.copy()
    offset = REGION_TIME_OFFSETS.get(region_name, 0)
    out["local_hour"] = (out["HR"] + offset) % 24
    month_shift = ((out["HR"] + offset) // 24).astype(int)
    out["local_month"] = out["MO"]
    out["local_month"] = (((out["local_month"] - 1 + month_shift) % 12) + 1).astype(int)
    out["sin_hour"] = np.sin(2 * np.pi * out["local_hour"] / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * out["local_hour"] / 24.0)
    out["sin_month"] = np.sin(2 * np.pi * out["local_month"] / 12.0)
    out["cos_month"] = np.cos(2 * np.pi * out["local_month"] / 12.0)
    return out

def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for lag in range(1, LOOKBACK + 1):
        out[f"{TARGET_COL}_lag_{lag}"] = out[TARGET_COL].shift(lag)
    for feature in BASE_FEATURES:
        for lag in [1, 2, 3, 6, 12, 24]:
            out[f"{feature}_lag_{lag}"] = out[feature].shift(lag)
    rolling_sources = [TARGET_COL, "CLOUD_AMT", "CLRSKY_SFC_SW_DWN"]
    for feature in rolling_sources:
        shifted = out[feature].shift(1)
        out[f"{feature}_roll_mean_3"] = shifted.rolling(window=3, min_periods=3).mean()
        out[f"{feature}_roll_mean_6"] = shifted.rolling(window=6, min_periods=6).mean()
        out[f"{feature}_roll_min_6"] = shifted.rolling(window=6, min_periods=6).min()
        out[f"{feature}_roll_max_6"] = shifted.rolling(window=6, min_periods=6).max()
        out[f"{feature}_roll_std_3"] = shifted.rolling(window=3, min_periods=3).std()
        out[f"{feature}_roll_std_6"] = shifted.rolling(window=6, min_periods=6).std()
    out[f"{TARGET_COL}_diff_1"] = out[TARGET_COL].shift(1) - out[TARGET_COL].shift(2)
    out[f"{TARGET_COL}_diff_3"] = out[TARGET_COL].shift(1) - out[TARGET_COL].shift(4)
    for feature in BASE_FEATURES:
        out[f"{feature}_diff_1"] = out[feature].shift(1) - out[feature].shift(2)
        out[f"{feature}_diff_3"] = out[feature].shift(1) - out[feature].shift(4)
    out["cloud_clearsky_interaction"] = out["CLOUD_AMT"].shift(1) * out["CLRSKY_SFC_SW_DWN"].shift(1)
    out["humidity_temp_interaction"] = out["RH2M"].shift(1) * out["T2M"].shift(1)
    return out


# ==========================================
# TASK 1: HEALTH CHECK ENDPOINT
# ==========================================
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "app_name": "Solar Irradiance Forecast API",
        "version": "0.1.0",
        "environment": "development"
    }

# ==========================================
# TASK 2: REGION LIST ENDPOINT
# ==========================================
@app.get("/regions")
def get_regions():
    regions_list = []
    for name, coords in COORDINATES.items():
        regions_list.append({
            "name": name,
            "folder_name": name.replace(" ", "_"), # <-- Tambahkan replace di sini
            "latitude": coords["lat"],
            "longitude": coords["lon"]
        })
    return {"regions": regions_list}

# ==========================================
# TASK 3: FORECAST ENDPOINT
# ==========================================
@app.post("/forecast")
async def get_forecast(request: ForecastRequest, db: Session = Depends(get_db)):
    region = request.region_name
    if region not in COORDINATES:
        raise HTTPException(status_code=404, detail="Wilayah tidak ditemukan")
        
    coords = COORDINATES[region]
    
    try:
        # --- PERBAIKAN BUG SPASI DI SINI ---
        folder_name = region.replace(" ", "_")
        base_dir = f"artifacts_xgboost_representative_points(Gen0.8)/{folder_name}"
        # -----------------------------------
        
        model_path = f"{base_dir}/xgboost_models.joblib"
        prep_path = f"{base_dir}/preprocessor.joblib"
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model AI untuk {region} tidak ditemukan di path: {model_path}")
            
        models_24_hours = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)
        feature_cols = preprocessor["feature_columns"]
        medians = preprocessor["train_feature_medians"]
        
        # referensi OpenMeteo
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&past_days=2&forecast_days=2&hourly=shortwave_radiation,cloud_cover,temperature_2m,relative_humidity_2m,surface_pressure,direct_radiation&timezone=Asia%2FJakarta"
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                raise ValueError("Gagal menyedot data dari Open-Meteo")
            om_data = resp.json()

        df = pd.DataFrame(om_data["hourly"])
        df["datetime"] = pd.to_datetime(df["time"])
        df["ALLSKY_SFC_SW_DWN"] = df["shortwave_radiation"]
        df["CLOUD_AMT"] = df["cloud_cover"]
        df["T2M"] = df["temperature_2m"]
        df["RH2M"] = df["relative_humidity_2m"]
        df["PS"] = df["surface_pressure"] / 10 
        df["CLRSKY_SFC_SW_DWN"] = df["direct_radiation"] 
        df["HR"] = df["datetime"].dt.hour
        df["MO"] = df["datetime"].dt.month
        
        df_processed = create_time_features(df, region)
        df_processed = create_lag_features(df_processed)
        
        now_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:00")
        current_rows = df_processed[df_processed["time"].str.startswith(now_str)]
        
        if current_rows.empty:
            last_row = df_processed.iloc[[-1]].copy()
            current_index = len(df_processed) - 24 # Fallback
        else:
            last_row = current_rows.copy()
            current_index = current_rows.index[0]
            
        for col in feature_cols:
            if col not in last_row.columns:
                last_row[col] = np.nan
        last_row[feature_cols] = last_row[feature_cols].fillna(medians)
        
        X_input = last_row[feature_cols].to_numpy(dtype=np.float32)
        
        forecast_hours = []
        model_prediction = []
        openmeteo_reference = []
        
        now = datetime.datetime.now()
        
        for i, model in enumerate(models_24_hours):
            target_time = now + datetime.timedelta(hours=i+1)
            iso_time = target_time.strftime("%Y-%m-%dT%H:00:00")
            forecast_hours.append(iso_time)
            
            # Prediksi
            pred = model.predict(X_input)[0]
            current_hour = target_time.hour
            final_pred = 0.0 if (current_hour >= 18 or current_hour <= 5) else max(0.0, float(pred))
            model_prediction.append(round(final_pred, 4))
            
            # Referensi OpenMeteo murni
            try:
                om_ref = float(df.iloc[current_index + i + 1]["shortwave_radiation"])
            except:
                om_ref = 0.0
            openmeteo_reference.append(round(om_ref, 4))
            
            # Simpan ke SQLite
            db_log = models.ForecastLog(
                region_name=region,
                forecast_time=iso_time,
                model_prediction=round(final_pred, 4)
            )
            db.add(db_log)
            
        db.commit()

        return {
            "region_name": region,
            "model_version": "xgboost_models.joblib",
            "generated_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "forecast_hours": forecast_hours,
            "model_prediction": model_prediction,
            "openmeteo_reference": openmeteo_reference,
            "source_weather": "Open-Meteo forecast API"
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# TASK 4: GET ALL LOGS
# ==========================================
@app.get("/logs")
def get_logs(limit: int = 100, db: Session = Depends(get_db)):
    logs = db.query(models.ForecastLog).order_by(models.ForecastLog.id.desc()).limit(limit).all()
    items = []
    for log in logs:
        items.append({
            "id": log.id,
            "region_name": log.region_name,
            "forecast_time": log.forecast_time,
            "model_prediction": log.model_prediction
        })
    return {
        "total": len(items),
        "items": items
    }

# ==========================================
# TASK 5: LOGS BY REGION
# ==========================================
@app.get("/logs/{region_name}")
def get_logs_by_region(region_name: str, limit: int = 100, db: Session = Depends(get_db)):
    if region_name not in COORDINATES:
        raise HTTPException(status_code=404, detail="Region not found")
        
    logs = db.query(models.ForecastLog).filter(models.ForecastLog.region_name == region_name).order_by(models.ForecastLog.id.desc()).limit(limit).all()
    
    items = []
    for log in logs:
        items.append({
            "id": log.id,
            "region_name": log.region_name,
            "forecast_time": log.forecast_time,
            "model_prediction": log.model_prediction
        })
        
    return {
        "total": len(items),
        "items": items
    }