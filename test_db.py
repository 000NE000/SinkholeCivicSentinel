from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()   # 프로젝트 루트의 .env 파일을 자동으로 읽어들임

# 2) 환경변수 읽기
PG_USER     = os.getenv("POSTGRES_USER")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
PG_DB       = os.getenv("POSTGRES_DB")
PG_PORT     = os.getenv("POSTGRES_PORT", "5432")  # 기본값 지정 가능
PG_HOST     = os.getenv("POSTGRES_HOST", "localhost")

print(PG_USER, PG_PASSWORD, PG_HOST, PG_PORT, PG_DB)

# 3) DSN 조합
DB_DSN = (
    f"postgresql://{PG_USER}:{PG_PASSWORD}"
    f"@{PG_HOST}:{PG_PORT}/{PG_DB}"
)

print(DB_DSN)

from src.etl.extract_api.ug_safety import get_subsidence_accident_list
from src.utils.config import API_CONFIG

lst = get_subsidence_accident_list(
    api_key=API_CONFIG["DATAGOKR_ENCODING_KEY"],
    sagoDateFrom="20240101",
    sagoDateTo="20241231",
    pageNo=1,
    numOfRows=5
)
print(lst.columns)
print(lst[["sagoNo", "sido", "sigungu"]].head())