import os
import asyncio
import csv
import psycopg2
import requests
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from src.utils.config import GOOGLE_CLOUD_PLATFORM_API

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()
GOOGLE_KEY = GOOGLE_CLOUD_PLATFORM_API.get("API_KEY")
if not GOOGLE_KEY:
    raise RuntimeError("Set GOOGLE_MAPS_API_KEY in your environment")

DB_DSN = os.getenv("DB_DSN")
if not DB_DSN:
    raise RuntimeError("Set DB_DSN in your environment")

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

# ---------------------------------------------------------------------------
# Ensure Table Exists and Clear Data
# ---------------------------------------------------------------------------
def prepare_table(conn):
    create_sql = '''
    CREATE TABLE IF NOT EXISTS survey_zone_geocoded (
        id SERIAL PRIMARY KEY,
        district TEXT,
        project_name TEXT,
        survey_section TEXT,
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        geom GEOMETRY(Point, 5179)
    );
    '''
    with conn.cursor() as cur:
        cur.execute(create_sql)
        # Clear existing data for overwrite
        cur.execute("TRUNCATE TABLE survey_zone_geocoded RESTART IDENTITY;")
    conn.commit()

# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------
def geocode_google(address: str):
    params = {"address": address, "key": GOOGLE_KEY, "language": "ko"}
    try:
        resp = requests.get(GEOCODE_URL, params=params, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error requesting {address}: {e}")
        return None
    data = resp.json()
    if data.get("status") != "OK" or not data.get("results"):
        print(f"No results for {address}: {data.get('status')}")
        return None
    loc = data["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]  # returns WGS84 lat, lon

async def async_geocode(loop, address):
    return await loop.run_in_executor(None, geocode_google, address)

# ---------------------------------------------------------------------------
# DB insert
# ---------------------------------------------------------------------------
INSERT_SQL = '''
    INSERT INTO survey_zone_geocoded (
        district, project_name, survey_section,
        latitude, longitude, geom, radius_km
    ) VALUES %s;
'''
TEMPLATE = (
    "(%s, %s, %s, %s, %s, %s, "
    "ST_SetSRID(ST_Buffer(ST_Point(%s, %s)::geography, %s*1000)::geometry, 5179))"
)

def insert_rows(conn, rows):
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(cur, INSERT_SQL, rows, template=TEMPLATE)
    conn.commit()

# ---------------------------------------------------------------------------
# CSV 읽고 비동기 처리
# ---------------------------------------------------------------------------
async def main():
    loop = asyncio.get_event_loop()

    # CSV 로드: 필요한 컬럼만 추출
    inputs = []
    with open("/Users/macforhsj/Desktop/SinkholeCivicSentinel/data/raw/seoul_construction_data.csv", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sec = row["survey_section"].split(" - ")[0].strip()
            inputs.append({
                "district":       row["district"],
                "project_name":   row["project_name"],
                "survey_section": row["survey_section"],
                "address":        sec,
                "radius_km": row["survey_length_km"],
            })

    # DB 연결 및 테이블 준비(생성+초기화)
    conn = psycopg2.connect(DB_DSN)
    prepare_table(conn)

    total = len(inputs)
    for idx, info in enumerate(inputs, 1):
        full_addr = f"{info['address']}, {info['district']}, 서울특별시"
        print(f"[{idx}/{total}] Geocoding: {full_addr}")
        latlon = await async_geocode(loop, full_addr)
        if latlon is None:
            print(f"  → Failed to geocode {full_addr}")
            continue
        lat, lon = latlon
        print(f"  → Got lat={lat}, lon={lon} (WGS84, transformed to SRID 5179)")

        row = (
            info["district"],
            info["project_name"],
            info["survey_section"],
            lat,
            lon,
            lon,
            lat
        )
        insert_rows(conn, [row])

    conn.close()
    print(f"✅ Done: processed {total} rows")

if __name__ == "__main__":
    asyncio.run(main())

