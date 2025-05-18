from sqlalchemy import create_engine
import requests
import asyncio
import json
import os
from dotenv import load_dotenv
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from src.utils.config import GPT_API, GOOGLE_CLOUD_PLATFORM_API

# --- Load configuration ----------------------------------------------------
load_dotenv()
client = AsyncOpenAI(api_key=GPT_API["API_KEY"])



# --------------------------------------------------------------------------
# 2. Google Maps Geocoding 호출 함수 (동기)
#    requests를 사용해 실시간 위/경도 데이터를 가져옵니다.
# --------------------------------------------------------------------------
def geocode_google(addr: str) -> tuple[float, float] | None:
    """
    Sync call to Google Maps Geocoding API.
    Returns (lat, lon) or None on failure.
    """
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": addr, "key": GOOGLE_CLOUD_PLATFORM_API["API_KEY"]}
    resp = requests.get(url, params=params, timeout=5)
    if resp.status_code != 200:
        return None

    data = resp.json()
    results = data.get("results")
    if not results:
        return None

    loc = results[0]["geometry"]["location"]
    return loc["lat"], loc["lng"]

# --------------------------------------------------------------------------
# 3. 비동기-동기 브리징 (Async ↔ Sync)
#    requests.get을 이벤트 루프에 블록킹 없이 실행하도록 run_in_executor 사용
# --------------------------------------------------------------------------
async def async_geocode(addr: str) -> tuple[float, float] | None:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, geocode_google, addr)

import os
import pandas as pd
import asyncio
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# --------------------------------------------------------------------------
# 4. 메인 워크플로우
#    CSV 로드 → 주소 조합 → 지오코딩 → Upsert → 진행상황 로깅
# --------------------------------------------------------------------------
async def main():
    # DB 커넥션 생성
    load_dotenv()
    DB_DSN = os.getenv("DB_DSN")
    engine = create_engine(DB_DSN, echo=False)

    # 4-1) CSV 로드 및 총 건수 계산
    df = pd.read_csv(
        "/Users/macforhsj/Desktop/SinkholeCivicSentinel/data/raw/"
        "citizen_reports/서울시 포트홀 보수 위치 정보.csv"
    )
    total = len(df)
    success = failed = 0

    # 4-2) 한 건씩 처리하며 진행 상황 출력
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        sagono       = row["등록번호"]     # 고유 ID
        area         = row["행정구역"]     # ex. "성북구"
        road_addr    = row["도로명주소"]   # 도로명주소
        damage_cat   = row["파손종류"]     # 파손종류

        # 4-3) 충분한 컨텍스트로 주소 문자열 생성
        full_addr = f"{area} {road_addr}"

        # 4-4) Google Maps API로 지오코딩 (비동기 실행)
        result = await async_geocode(full_addr)
        if not result:
            failed += 1
            print(f"[{idx}/{total}] ❌ {sagono}: Geocode 실패 for '{full_addr}'")
            continue
        lat, lon = result

        # 4-5) PostGIS에 Upsert (INSERT … ON CONFLICT DO UPDATE)
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO porthole_report (
                      sagono,
                      damage_category,
                      geom
                    ) VALUES (
                      :sagono,
                      :damage_category,
                      ST_SetSRID(ST_Point(:lon, :lat), 5179)
                    )
                    ON CONFLICT (sagono) DO UPDATE
                      SET geom            = EXCLUDED.geom,
                          damage_category = EXCLUDED.damage_category;
                """),
                {
                    "sagono":        sagono,
                    "damage_category": damage_cat,
                    "lon":           lon,
                    "lat":           lat
                }
            )

        success += 1
        print(f"[{idx}/{total}] ✅ {sagono}: ({lat:.6f}, {lon:.6f})")

    # 4-6) 최종 요약
    print(f"\n완료: 총={total}, 성공={success}, 실패={failed}")

if __name__ == "__main__":
    asyncio.run(main())



