import os
import sys
import sqlparse
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_DSN = os.getenv("DB_DSN")            # must be set
DDL_PATH = "/Users/macforhsj/Desktop/SinkholeCivicSentinel/db/schema.sql"              # relative path to your SQL file

def execute_sql(db_dsn: str, ddl_path: str) -> None:
    # 0. 파일 존재 확인
    if not os.path.isfile(ddl_path):
        print(f"ERROR: DDL file not found at {ddl_path}", file=sys.stderr)
        sys.exit(1)

    # 1. SQL 읽기
    with open(ddl_path, "r", encoding="utf-8") as f:
        raw_sql = f.read()

    # 2. sqlparse로 문장 단위 분리
    statements = sqlparse.split(raw_sql)

    # 3. DB 연결
    try:
        conn = psycopg2.connect(db_dsn)
    except Exception as e:
        print(f"ERROR: Could not connect to database:\n  {e}", file=sys.stderr)
        sys.exit(1)

    # 4. 커서로 문장별 실행
    with conn:
        with conn.cursor() as cur:
            for idx, stmt in enumerate(statements, start=1):
                stmt = stmt.strip()
                if not stmt:
                    continue
                try:
                    cur.execute(stmt)
                    conn.commit()  # 문장별 커밋
                    print(f"[OK]   Statement #{idx} executed.")
                except Exception as e:
                    conn.rollback()
                    print(f"[FAIL] Statement #{idx} failed:\n  {e}\n  SQL: {stmt[:200]}...", file=sys.stderr)
    conn.close()
    print("Schema applied (with individual commits).")

if __name__ == "__main__":
    if not DB_DSN:
        raise RuntimeError("Missing DB_DSN in environment")
    execute_sql(DB_DSN, DDL_PATH)