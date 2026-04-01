from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. Tentukan nama file database SQLite-nya
SQLALCHEMY_DATABASE_URL = "sqlite:///./solar_telemetry.db"

# 2. Buat "Mesin" penghubung (check_same_thread=False wajib untuk FastAPI + SQLite)
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# 3. Buat pabrik sesi (Session) untuk ngobrol dengan database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. Buat fondasi dasar untuk tabel-tabel kita nanti
Base = declarative_base()