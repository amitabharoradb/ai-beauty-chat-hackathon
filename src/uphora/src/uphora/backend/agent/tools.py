import os

from databricks import sql as dbsql
from databricks.sdk import WorkspaceClient

UC_CATALOG = os.getenv("UC_CATALOG", "amitabh_arora_catalog")
UC_SCHEMA = os.getenv("UC_SCHEMA", "uphora_hackathon")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "9465acf928ae5952")

def _query_products(where_clause: str = "1=1", limit: int = 10) -> list[dict]:
    """Query products from Unity Catalog via Databricks SQL connector (uses env/profile auth)."""
    w = WorkspaceClient()
    with dbsql.connect(
        server_hostname=w.config.host.replace("https://", ""),
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        access_token=w.config.token,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, category_id, name, description, price,
                       key_ingredients, benefits, tags
                FROM {UC_CATALOG}.{UC_SCHEMA}.products
                WHERE {where_clause}
                LIMIT {limit}
            """)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    return [dict(zip(cols, row)) for row in rows]


def search_products(
    query: str,
    category: str | None,
    filters: dict,
    limit: int = 6,
) -> list[dict]:
    """
    Search products by query text and optional filters.
    filters: {"vegan": bool, "fragrance_free": bool, "max_price": float}
    """
    conditions = ["1=1"]

    if category:
        cat_map = {"skincare": "cat_skincare", "makeup": "cat_makeup", "haircare": "cat_haircare"}
        cat_id = cat_map.get(category.lower())
        if cat_id:
            conditions.append(f"category_id = '{cat_id}'")

    if filters.get("vegan"):
        conditions.append("array_contains(tags, 'vegan')")

    if filters.get("fragrance_free"):
        conditions.append("array_contains(tags, 'fragrance-free')")

    if filters.get("max_price"):
        conditions.append(f"price <= {float(filters['max_price'])}")

    if query:
        safe_query = query.replace("'", "''")
        conditions.append(
            f"(lower(name) LIKE '%{safe_query.lower()}%' "
            f"OR lower(description) LIKE '%{safe_query.lower()}%')"
        )

    products = _query_products(" AND ".join(conditions), limit=limit)
    return products


def get_routine(memory: dict) -> dict:
    """Extract AM/PM routine from customer memory."""
    routines = memory.get("routines", {})
    return {
        "am": routines.get("am", []),
        "pm": routines.get("pm", []),
    }
