import pandas as pd

# 1) CSV 로드
df = pd.read_csv(
    "/Users/macforhsj/Desktop/SinkholeCivicSentinel/data/raw/하수관거통계_cleaned.csv",
    encoding="utf-8"
)

# 2) Wide → Long 변환 (melt)
# 2-1) change
change_cols = [c for c in df.columns if c.startswith("change_")]
m1 = df.melt(
    id_vars="district_id",
    value_vars=change_cols,
    var_name="orig_period",
    value_name="change_count"
)
# 2-2) install
inst_cols = [c for c in df.columns if c.startswith("install_rate_")]
m2 = df.melt(
    id_vars="district_id",
    value_vars=inst_cols,
    var_name="orig_period",
    value_name="install_rate"
)
# 2-3) replace
rep_cols = [c for c in df.columns if c.startswith("replacement_rate_")]
m3 = df.melt(
    id_vars="district_id",
    value_vars=rep_cols,
    var_name="orig_period",
    value_name="replace_rate"
)

# 3) 공통의 period 키 생성: "YYYY_YYYY"
for dfm in (m1, m2, m3):
    dfm["period"] = dfm["orig_period"].str.extract(r"(\d{4}_\d{4})")
    dfm.drop(columns="orig_period", inplace=True)

# 4) merge
long_df = (
    m1
    .merge(m2, on=["district_id", "period"])
    .merge(m3, on=["district_id", "period"])
)

# 5) period → year_start, year_end
year_cols = long_df["period"].str.extract(r"(\d{4})_(\d{4})")
year_cols.columns = ["year_start", "year_end"]
long_df = pd.concat([long_df.drop(columns="period"), year_cols], axis=1)

# 6) 타입 변환 & 결측 처리
long_df = long_df.astype({
    "district_id": "int64",
    "year_start": "int64",
    "year_end": "int64",
    "change_count": "Int64",
    "install_rate": "float64",
    "replace_rate": "float64"
}).fillna(0)

# 7) district_id 0부터 차례대로 매핑할 '구 이름' 리스트
district_names = [
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구",
    "중랑구", "성북구", "강북구", "도봉구", "노원구", "은평구",
    "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구",
    "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구",
    "강동구"
]

# 8) 매핑용 DataFrame 생성
name_map = pd.DataFrame({
    "district_id": list(range(len(district_names))),
    "district_name": district_names
})

# 9) 병합
long_df = long_df.merge(name_map, on="district_id", how="left")

# 10) 결과 확인
print(long_df.head(10))
print("총 행 개수:", len(long_df))

# 10) (선택) 데이터베이스 적재 예시
from sqlalchemy import create_engine, Integer, BigInteger, Float
from dotenv import load_dotenv
import os

load_dotenv()
db_dsn = os.getenv("DB_DSN")
engine = create_engine(db_dsn)

long_df.to_sql(
    "sewer_district_yearly",
    engine,
    if_exists="replace",
    index=False,
    dtype={
        "district_id": Integer(),
        "year_start": Integer(),
        "year_end": Integer(),
        "change_count": BigInteger(),
        "install_rate": Float(),
        "replace_rate": Float(),
    }
)