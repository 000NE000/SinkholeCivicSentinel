import os
import asyncio
import json
import psycopg2
from threading import Lock
from dotenv import load_dotenv
from openai import AsyncOpenAI
from psycopg2.extras import execute_values
from src.utils.config import GPT_API

# --- Load & init ------------------------------------------------------------
load_dotenv()
client = AsyncOpenAI(api_key=GPT_API["API_KEY"])

# --- Rate limiter (async) ---------------------------------------------------
class TokenBucket:
    """Simple async‑safe token bucket for OpenAI rate limiting."""
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.timestamp = asyncio.get_event_loop().time()
        self.lock = Lock()

    def consume(self, amount: int = 1) -> bool:
        """Return True if tokens are available and immediately consume."""
        with self.lock:
            now = asyncio.get_event_loop().time()
            self.tokens = min(
                self.capacity,
                self.tokens + (now - self.timestamp) * self.rate,
            )
            self.timestamp = now
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False

bucket = TokenBucket(rate=1, capacity=1)  # 60 RPM limit → 1 rps conservative

# --- Function schema & geocode funcs ---------------------------------------
FUNCTIONS = [
    {
        "name": "get_geocode_full",
        "description": "Return full parsing of an address, including all components and candidate list",
        "parameters": {
            "type": "object",
            "properties": {
                "candidate_addresses": {"type": "string"},
                "sido": {"type": ["string", "null"]},
                "sigungu": {"type": ["string", "null"]},
                "dong": {"type": ["string", "null"]},
                "addr": {"type": ["string", "null"]},
                "addr_detail": {"type": ["string", "null"]},
                "latitude": {"type": ["number", "null"]},
                "longitude": {"type": ["number", "null"]},
            },
            "required": ["candidate_addresses", "latitude", "longitude"],
        },
    }
]

async def geocode_openai_full(query: str):
    """Call GPT‑4o once per query and return parsed dict."""
    while not bucket.consume():
        await asyncio.sleep(0.1)

    prompt = (
        "You are a geocoding assistant. Given the address query:\n\n"
        f"    {query}\n\n"
        "Return a JSON object with these fields:\n"
        "  • candidate_addresses: raw candidates JSON string\n"
        "  • sido, sigungu, dong: 행정 구역\n"
        "  • addr: street + number\n"
        "  • addr_detail: building name etc.\n"
        "  • latitude, longitude (float)\n"
        "Return null for missing values."
    )

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You parse geocoding results like Naver."},
            {"role": "user", "content": prompt},
        ],
        functions=FUNCTIONS,
        function_call={"name": "get_geocode_full"},
        temperature=0.0,
    )

    msg = response.choices[0].message
    if msg.function_call:
        return json.loads(msg.function_call.arguments), False
    return None, True

async def safe_geocode(query: str):
    if not query or query.strip() in {"-", "--"}:
        return None, True
    return await geocode_openai_full(query)

# --- Parse & collect rows --------------------------------------------------
def parse_address(sagono: str, data: dict):
    """Convert GPT response → tuple aligned with SQL template (12 elements)."""
    return (
        sagono,
        data.get("candidate_addresses", "[]"),
        data.get("sido"),
        data.get("sigungu"),
        data.get("dong"),
        data.get("addr"),
        data.get("addr"),                # standard_address
        data.get("addr_detail"),
        data.get("latitude"),
        data.get("longitude"),
        data.get("longitude"),            # geom lon
        data.get("latitude"),             # geom lat
    )

# --- Database Upsert -------------------------------------------------------
def upsert_addresses(conn: psycopg2.extensions.connection, rows: list[tuple]):
    """Batch upsert; swallow errors, keep going."""
    sql = """
        INSERT INTO subsidence_accident_info (
            sagono, candidate_addresses,
            sido, sigungu, dong, addr, standard_address, addr_detail,
            latitude, longitude, geom
        ) VALUES %s
        ON CONFLICT (sagono) DO UPDATE SET
            candidate_addresses = EXCLUDED.candidate_addresses,
            sido                = EXCLUDED.sido,
            sigungu             = EXCLUDED.sigungu,
            dong                = EXCLUDED.dong,
            addr                = EXCLUDED.addr,
            standard_address    = EXCLUDED.standard_address,
            addr_detail         = EXCLUDED.addr_detail,
            latitude            = EXCLUDED.latitude,
            longitude           = EXCLUDED.longitude,
            geom                = EXCLUDED.geom;
    """

    # 10 placeholders for columns BEFORE geom + 2 for lon/lat in ST_Point
    template = (
        "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
        "ST_SetSRID(ST_Point(%s, %s), 4326))"
    )

    try:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, template=template)
        conn.commit()
        print(f"DEBUG upsert success; {len(rows)} rows")
    except Exception as e:
        conn.rollback()
        print(f"DEBUG upsert error for {[r[0] for r in rows]}: {e}")

# --- Async main ------------------------------------------------------------

async def main():
    # Load inputs
    with psycopg2.connect(os.environ["DB_DSN"]) as read_conn:
        with read_conn.cursor() as cur:
            cur.execute(
                "SELECT sagono, CONCAT_WS(' ', sido, sigungu, dong, addr) FROM subsidence_accident_info;"
            )
            inputs = cur.fetchall()

    upsert_conn = psycopg2.connect(os.environ["DB_DSN"])

    total = len(inputs)
    review = []

    for idx, (sagono, query) in enumerate(inputs, 1):
        print(f"[{idx}/{total}] Geocoding: {query}")
        data, skip = await safe_geocode(query)

        if skip:
            row = (sagono, "[]", None, None, None, None, None, None, None, None, None, None)
            review.append(sagono)
        else:
            row = parse_address(sagono, data)
            if data.get("latitude") is None:
                review.append(sagono)

        upsert_addresses(upsert_conn, [row])

    upsert_conn.close()
    print(f"✅ Done: {total} rows; needs review: {len(review)} → {review}")

if __name__ == "__main__":
    asyncio.run(main())
