# app/utils.py
import logging
import os
import json
from google.cloud import bigquery   # ✅ needed for read_from_bigquery

# Create a module-level logger for utils
logger = None


def get_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Creates a logger with both console and optional file handler.
    - name: name of the logger (usually __name__ or module name)
    - log_file: if provided, logs are written to this file as well
    - level: logging level (default: INFO)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:  # prevent duplicate handlers if reimported
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger


# ✅ Initialize a default logger for utils
logger = get_logger("Utils", log_file="logs/utils.log")


def save_json(data, file_name="requirements.json"):
    """Save Python object as a formatted JSON file."""
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved {len(data)} records locally → {file_name}")
    except Exception as e:
        logger.error(f"❌ Failed to save JSON {file_name}: {e}")


def read_json(file_path="requirements.json"):
    """Load requirements from a local JSON file."""
    if not os.path.exists(file_path):
        logger.warning(f"⚠️ JSON file not found → {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"✅ Loaded {len(data)} requirements from {file_path}")
        return data
    except Exception as e:
        logger.error(f"❌ Failed to read JSON file {file_path}: {e}")
        return []


def read_from_bigquery(project_id, dataset_id, table_id="requirements", limit=None):
    """Load requirements from BigQuery table."""
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    query = f"SELECT * FROM `{table_ref}`"
    if limit:
        query += f" LIMIT {limit}"

    try:
        logger.info(f"📥 Querying BigQuery table: {table_ref}")
        query_job = client.query(query)
        rows = [dict(row) for row in query_job.result()]
        logger.info(f"✅ Loaded {len(rows)} rows from {table_ref}")
        return rows
    except Exception as e:
        logger.error(f"❌ Failed to load from BigQuery {table_ref}: {e}")
        return []
