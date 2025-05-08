#!/usr/bin/env python3
import os
import sys
import logging
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from datetime import date

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Load .env from the project root (adjust if this script lives elsewhere)
env_path = Path("/Users/macforhsj/Desktop/SinkholeCivicSentinel/.env")
load_dotenv(dotenv_path=env_path)

# Environment variables
PG_USER     = os.getenv('POSTGRES_USER')
PG_PASSWORD = os.getenv('POSTGRES_PASSWORD')
PG_DB       = os.getenv('POSTGRES_DB')
PG_PORT     = os.getenv('POSTGRES_PORT', '5432')
PG_HOST     = os.getenv('PG_HOST', 'localhost')
BACKUP_DIR  = Path(os.getenv('BACKUP_DIR', './backups'))
DATE_STR    = date.today().isoformat()
BACKUP_FILE = BACKUP_DIR / f"{PG_DB}_{DATE_STR}.dump"

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

# -----------------------------------------------------------------------------
# Ensure backup directory exists
# -----------------------------------------------------------------------------
try:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Using backup directory: {BACKUP_DIR}")
except Exception as e:
    logging.error(f"Failed to create backup directory: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Construct pg_dump command
# -----------------------------------------------------------------------------
cmd = [
    'pg_dump',
    '-h', PG_HOST,
    '-p', PG_PORT,
    '-U', PG_USER,
    '-F', 'c',            # custom format
    '-f', str(BACKUP_FILE),
    PG_DB
]

# PG password via environment
env = os.environ.copy()
env['PGPASSWORD'] = PG_PASSWORD

# -----------------------------------------------------------------------------
# Execute backup
# -----------------------------------------------------------------------------
try:
    logging.info(f"Starting backup: {BACKUP_FILE.name}")
    subprocess.run(cmd, check=True, env=env)
    logging.info(f"Backup completed successfully: {BACKUP_FILE}")
except subprocess.CalledProcessError as e:
    logging.error(f"pg_dump failed with exit code {e.returncode}")
    sys.exit(e.returncode)
except Exception as e:
    logging.error(f"Unexpected error during backup: {e}")
    sys.exit(1)