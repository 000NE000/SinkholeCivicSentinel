import os
import asyncio
import json
import psycopg2
import requests
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from src.utils.config import GOOGLE_CLOUD_PLATFORM_API

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()
GOOGLE_KEY = GOOGLE_CLOUD_PLATFORM_API["API_KEY"]

if not GOOGLE_KEY:
    raise RuntimeError("Set GOOGLE_MAPS_API_KEY in your environment")
DB_DSN = os.getenv("DB_DSN")
if not DB_DSN:
    raise RuntimeError("Set DB_DSN in your environment")

# ---------------------------------------------------------------------------
# Google Maps Geocoding
# ---------------------------------------------------------------------------
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


def geocode_google(address: str):
    """Return (lat, lon) tuple or None using Google Geocoding API."""
    params = {"address": address, "key": GOOGLE_KEY, "language": "ko"}
    try:
        resp = requests.get(GEOCODE_URL, params=params, timeout=5)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    data = resp.json()
    if data.get("status") != "OK" or not data.get("results"):
        return None
    loc = data["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]


async def async_geocode(loop, address):
    """Run blocking geocode_google in executor to avoid blocking event loop."""
    return await loop.run_in_executor(None, geocode_google, address)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
INSERT_SQL = """
    INSERT INTO subsidence_accident_info (
        sagono, candidate_addresses,
        sido, sigungu, dong, addr, standard_address, addr_detail,
        latitude, longitude, geom
    ) VALUES %s
    ON CONFLICT (sagono) DO UPDATE SET
        latitude = EXCLUDED.latitude,
        longitude = EXCLUDED.longitude,
        geom     = EXCLUDED.geom;
"""

TEMPLATE = (
    "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
    "ST_SetSRID(ST_Point(%s, %s), 5179))"
)


def upsert_rows(conn, rows):
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(cur, INSERT_SQL, rows, template=TEMPLATE)
    conn.commit()


# ---------------------------------------------------------------------------
# Main async workflow
# ---------------------------------------------------------------------------
async def main():
    loop = asyncio.get_event_loop()

    with psycopg2.connect(DB_DSN) as read_conn:
        with read_conn.cursor() as cur:
            cur.execute(
                "SELECT sagono, CONCAT_WS(' ', sido, sigungu, dong, addr) "
                "FROM subsidence_accident_info;"
            )
            inputs = cur.fetchall()

    upsert_conn = psycopg2.connect(DB_DSN)

    total = len(inputs)
    review = []

    for idx, (sagono, full_addr) in enumerate(inputs, 1):
        print(f"[{idx}/{total}] Geocoding: {full_addr}")
        latlon = await async_geocode(loop, full_addr)

        if latlon is None:
            # keep existing record, mark for review
            review.append(sagono)
            continue
        lat, lon = latlon
        row = (
            sagono,
            "[]",      # candidate_addresses placeholder
            None, None, None, None, None, None,
            lat,
            lon,
            lon,
            lat,
        )
        upsert_rows(upsert_conn, [row])

    upsert_conn.close()
    print(f"✅ Done: {total} rows; needs manual review: {len(review)} -> {review}")


if __name__ == "__main__":
    asyncio.run(main())