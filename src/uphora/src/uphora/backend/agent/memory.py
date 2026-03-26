import json
import os

try:
    import psycopg
except ImportError:
    psycopg = None

from databricks.sdk import WorkspaceClient

LAKEBASE_PROJECT_ID = os.getenv("LAKEBASE_PROJECT_ID", "uphora-hackathon-memory")
DB_NAME = "databricks_postgres"

def _get_conn():
    """Get a psycopg connection with fresh OAuth token."""
    w = WorkspaceClient()
    endpoints = list(w.postgres.list_endpoints(
        parent=f"projects/{LAKEBASE_PROJECT_ID}/branches/production"
    ))
    ep_name = endpoints[0].name
    endpoint = w.postgres.get_endpoint(name=ep_name)
    host = endpoint.status.hosts.host
    cred = w.postgres.generate_database_credential(endpoint=ep_name)
    username = w.current_user.me().user_name
    return psycopg.connect(
        host=host,
        dbname=DB_NAME,
        user=username,
        password=cred.token,
        sslmode="require",
    )


def _load_memory_impl(customer_id: str) -> dict:
    """Load long-term memory for a customer from Lakebase."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT skin_profile, goals, preferences, routines,
                       product_history, category_affinities
                FROM customer_memory
                WHERE customer_id = %s
            """, (customer_id,))
            row = cur.fetchone()
        if row is None:
            return {}
        return {
            "skin_profile":         row[0] if isinstance(row[0], dict) else json.loads(row[0]),
            "goals":                row[1] if isinstance(row[1], list) else json.loads(row[1]),
            "preferences":          row[2] if isinstance(row[2], dict) else json.loads(row[2]),
            "routines":             row[3] if isinstance(row[3], dict) else json.loads(row[3]),
            "product_history":      row[4] if isinstance(row[4], dict) else json.loads(row[4]),
            "category_affinities":  row[5] if isinstance(row[5], list) else json.loads(row[5]),
        }
    finally:
        conn.close()


def load_memory_sync(customer_id: str) -> dict:
    """Synchronous version — used by LangGraph nodes (ChatDatabricks is also sync)."""
    return _load_memory_impl(customer_id)

async def load_memory(customer_id: str) -> dict:
    """Async wrapper for non-graph callers."""
    return _load_memory_impl(customer_id)


def _save_memory_impl(customer_id: str, memory: dict, delta: dict) -> None:
    """Upsert memory delta back to Lakebase."""
    merged = {**memory, **delta}
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO customer_memory
                    (customer_id, skin_profile, goals, preferences, routines,
                     product_history, category_affinities, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (customer_id) DO UPDATE SET
                    product_history      = EXCLUDED.product_history,
                    routines             = EXCLUDED.routines,
                    category_affinities  = EXCLUDED.category_affinities,
                    updated_at           = NOW()
            """, (
                customer_id,
                json.dumps(merged.get("skin_profile", {})),
                json.dumps(merged.get("goals", [])),
                json.dumps(merged.get("preferences", {})),
                json.dumps(merged.get("routines", {})),
                json.dumps(merged.get("product_history", {})),
                json.dumps(merged.get("category_affinities", [])),
            ))
        conn.commit()
    finally:
        conn.close()


def save_memory_sync(customer_id: str, memory: dict, delta: dict) -> None:
    """Synchronous version — used by FastAPI router after predict_stream."""
    _save_memory_impl(customer_id, memory, delta)

async def save_memory(customer_id: str, memory: dict, delta: dict) -> None:
    _save_memory_impl(customer_id, memory, delta)


def _load_session_summaries_impl(customer_id: str) -> list[str]:
    """Load the 5 most recent session summaries."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT summary FROM session_summaries
                WHERE customer_id = %s
                ORDER BY created_at DESC
                LIMIT 5
            """, (customer_id,))
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

def load_session_summaries_sync(customer_id: str) -> list[str]:
    return _load_session_summaries_impl(customer_id)

async def load_session_summaries(customer_id: str) -> list[str]:
    return _load_session_summaries_impl(customer_id)


def _save_session_summary_impl(customer_id: str, session_id: str, summary: str) -> None:
    """Persist a session summary (keep last 5 per customer)."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO session_summaries (customer_id, session_id, summary)
                VALUES (%s, %s, %s)
            """, (customer_id, session_id, summary))
            cur.execute("""
                DELETE FROM session_summaries
                WHERE customer_id = %s
                  AND id NOT IN (
                      SELECT id FROM session_summaries
                      WHERE customer_id = %s
                      ORDER BY created_at DESC
                      LIMIT 5
                  )
            """, (customer_id, customer_id))
        conn.commit()
    finally:
        conn.close()

async def save_session_summary(customer_id: str, session_id: str, summary: str) -> None:
    _save_session_summary_impl(customer_id, session_id, summary)
