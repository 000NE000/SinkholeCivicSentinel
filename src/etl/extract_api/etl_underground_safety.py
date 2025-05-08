from src.etl.extract_api.ug_safety import get_subsidence_accident_list, get_subsidence_accident_info, get_small_impact_evaluation_list
from src.etl.extract_api.ug_safety import get_underground_utility_list
from src.utils.config import API_CONFIG
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import os
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from psycopg2.pool import ThreadedConnectionPool

# 1) .env 로드
load_dotenv()

# 2) DB 연결 정보
PG_USER     = os.getenv("POSTGRES_USER")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
PG_DB       = os.getenv("POSTGRES_DB")
PG_PORT     = os.getenv("POSTGRES_PORT", "5432")
PG_HOST     = os.getenv("POSTGRES_HOST", "localhost")
DB_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"

api_key = API_CONFIG["DATAGOKR_ENCODING_KEY"]

pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=DB_DSN)


def get_list(geomLon: float, geomLat: float, buffer: int):
    lst = get_small_impact_evaluation_list(
        api_key=api_key,
        sysRegDateFrom="20190101",
        sysRegDateTo="20250506",
        geomLon=geomLon,
        geomLat=geomLat,
        buffer=buffer,
    )
    lst = get_underground_utility_list(api_key, startYmd = 2019, endYmd = 2024)
    print(len(lst))
    print(lst.head())

def etl_accident_list():


    # 1) 사고 리스트만 조회 (entryYmdFrom/To 포함)
    lst = get_subsidence_accident_list(
        api_key=api_key,
        sagoDateFrom="20190101",
        sagoDateTo="20250506",
        pageNo=1,
        numOfRows=100
    )

    # 2) 필요한 컬럼만 추출해 리스트로 변환
    records = []
    for _, row in lst.iterrows():
        records.append((
            row["sagoNo"],
            row.get("siDo"),
            row.get("siGunGu"),
            row.get("sagoReason"),
            row.get("sagoDate"),
        ))

    # 3) DB 적재
    conn = psycopg2.connect(DB_DSN)
    cur  = conn.cursor()
    execute_values(cur, """
        INSERT INTO subsidence_accident_list
          (sagoNo, siDo, siGunGu, sagoReason, sagoDate)
        VALUES %s
        ON CONFLICT (sagoNo) DO UPDATE
          SET
            siDo       = EXCLUDED.siDo,
            siGunGu    = EXCLUDED.siGunGu,
            sagoReason = EXCLUDED.sagoReason,
            sagoDate   = EXCLUDED.sagoDate
    """, records)
    conn.commit()


    cur.close()
    conn.close()


def worker(sagoNo: str):
    conn = pool.getconn()
    try:
        df = get_subsidence_accident_info(api_key=api_key, sagoNo=sagoNo)
        df.columns = [c.lower() for c in df.columns]
        if df.empty:
            return

        row = df.iloc[0]
        # normalize columns to lowercase above
        rec = (
            row.get("sagono"),
            row.get("sido"),
            row.get("sigungu"),
            row.get("dong"),
            row.get("addr"),
            row.get("sagodate"),
            row.get("sinkwidth"),
            row.get("sinkextend"),
            row.get("sinkdepth"),
            row.get("grdkind"),
            row.get("sagoreason"),
            row.get("deathcnt"),
            row.get("injurycnt"),
            row.get("vehiclecnt"),
        )

        cur = conn.cursor()
        cur.execute("""
            INSERT INTO subsidence_accident_info (
              sagono, sido, sigungu, dong, addr,
              sagodate, sinkwidth, sinkextend, sinkdepth,
              grdkind, sagoreason, deathcnt, injurycnt, vehiclecnt
            ) VALUES (
              %s, %s, %s, %s, %s,
              %s, %s, %s, %s, %s,
              %s, %s, %s, %s
            )
            ON CONFLICT (sagono) DO UPDATE SET
              sido       = EXCLUDED.sido,
              sigungu    = EXCLUDED.sigungu,
              dong       = EXCLUDED.dong,
              addr       = EXCLUDED.addr,
              sagodate   = EXCLUDED.sagodate,
              sinkwidth  = EXCLUDED.sinkwidth,
              sinkextend = EXCLUDED.sinkextend,
              sinkdepth  = EXCLUDED.sinkdepth,
              grdkind    = EXCLUDED.grdkind,
              sagoreason = EXCLUDED.sagoreason,
              deathcnt   = EXCLUDED.deathcnt,
              injurycnt  = EXCLUDED.injurycnt,
              vehiclecnt = EXCLUDED.vehiclecnt;
        """, rec)
        conn.commit()
        cur.close()
        time.sleep(0.2)

    finally:
        pool.putconn(conn)


def etl_accident_info_parallel(max_workers: int = 8):
    conn = psycopg2.connect(DB_DSN)
    cur  = conn.cursor()
    cur.execute("SELECT sagoNo FROM subsidence_accident_list;")
    sago_nos = [r[0] for r in cur.fetchall()]
    cur.close(); conn.close()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, no): no for no in sago_nos}
        for _ in tqdm(as_completed(futures), total=len(futures),
                      desc="Accident Detail ETL"):
            pass

    pool.closeall()


if __name__ == "__main__":
    # etl_accident_list()
    # etl_accident_info()
    # etl_accident_info_parallel()
    get_list(geomLon=127.0, geomLat=37.55, buffer= 10000)
