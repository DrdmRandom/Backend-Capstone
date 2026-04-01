from sqlalchemy import Column, Integer, String, Float, DateTime
from database import Base
import datetime

class ForecastLog(Base):
    __tablename__ = "forecast_logs"

    id = Column(Integer, primary_key=True, index=True)
    region_name = Column(String, index=True)
    forecast_time = Column(String)
    model_prediction = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)