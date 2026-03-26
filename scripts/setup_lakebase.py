# scripts/setup_lakebase.py
"""
Creates Lakebase Autoscaling project, tables, and seeds 10 demo customers.
Run from local machine or Databricks driver node.
Usage: python scripts/setup_lakebase.py
"""
import json
import os
import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import Project, ProjectSpec

PROJECT_ID = os.getenv("LAKEBASE_PROJECT_ID", "uphora-hackathon-memory")
DB_NAME = "databricks_postgres"

w = WorkspaceClient()

# --- 1. Create project (idempotent) ---
try:
    project = w.postgres.get_project(name=f"projects/{PROJECT_ID}")
    print(f"Project already exists: {project.name}")
except Exception:
    print(f"Creating Lakebase project '{PROJECT_ID}'...")
    op = w.postgres.create_project(
        project=Project(spec=ProjectSpec(display_name="Uphora Memory", pg_version="17")),
        project_id=PROJECT_ID,
    )
    project = op.wait()
    print(f"Created: {project.name}")

# --- 2. Get connection details ---
endpoints = list(w.postgres.list_endpoints(
    parent=f"projects/{PROJECT_ID}/branches/production"
))
ep_name = endpoints[0].name
endpoint = w.postgres.get_endpoint(name=ep_name)
host = endpoint.status.hosts.host
cred = w.postgres.generate_database_credential(endpoint=ep_name)
username = w.current_user.me().user_name

conn_str = (
    f"host={host} dbname={DB_NAME} user={username} "
    f"password={cred.token} sslmode=require"
)

# --- 3. Create tables ---
CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS customer_memory (
    customer_id     TEXT PRIMARY KEY,
    skin_profile    JSONB NOT NULL DEFAULT '{}',
    goals           JSONB NOT NULL DEFAULT '[]',
    preferences     JSONB NOT NULL DEFAULT '{}',
    routines        JSONB NOT NULL DEFAULT '{}',
    product_history JSONB NOT NULL DEFAULT '{}',
    category_affinities JSONB NOT NULL DEFAULT '[]',
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS session_summaries (
    id          SERIAL PRIMARY KEY,
    customer_id TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    summary     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_session_summaries_customer
    ON session_summaries(customer_id, created_at DESC);
"""

# --- 4. Seed 10 demo customers ---
DEMO_CUSTOMERS = [
    {
        "customer_id": "cust_00001",
        "skin_profile": {"type": "oily", "tone": "medium", "concerns": ["acne", "enlarged pores"], "sensitivities": ["fragrance"]},
        "goals": ["clear skin", "glow"],
        "preferences": {"vegan": True, "fragrance_free": True, "budget_range": [20, 80]},
        "routines": {"am": ["cleanser", "toner", "moisturizer", "SPF"], "pm": ["cleanser", "serum", "moisturizer"]},
        "product_history": {"recommended": [], "liked": [], "disliked": []},
        "category_affinities": ["skincare", "makeup", "haircare"],
    },
    {
        "customer_id": "cust_00002",
        "skin_profile": {"type": "dry", "tone": "fair", "concerns": ["dryness", "redness"], "sensitivities": ["alcohol", "retinol"]},
        "goals": ["intense hydration", "calm redness"],
        "preferences": {"vegan": False, "fragrance_free": True, "budget_range": [30, 100]},
        "routines": {"am": ["gentle cleanser", "serum", "rich moisturizer", "SPF"], "pm": ["oil cleanser", "serum", "night cream"]},
        "product_history": {"recommended": [], "liked": [], "disliked": []},
        "category_affinities": ["skincare", "haircare", "makeup"],
    },
    {
        "customer_id": "cust_00003",
        "skin_profile": {"type": "combination", "tone": "light", "concerns": ["anti-aging", "hyperpigmentation"], "sensitivities": []},
        "goals": ["anti-aging", "even skin tone"],
        "preferences": {"vegan": True, "fragrance_free": False, "budget_range": [40, 120]},
        "routines": {"am": ["cleanser", "vitamin C", "moisturizer", "SPF"], "pm": ["cleanser", "retinol", "moisturizer"]},
        "product_history": {"recommended": [], "liked": [], "disliked": []},
        "category_affinities": ["skincare", "makeup", "haircare"],
    },
    {
        "customer_id": "cust_00004",
        "skin_profile": {"type": "sensitive", "tone": "deep", "concerns": ["redness", "dryness"], "sensitivities": ["fragrance", "essential oils"]},
        "goals": ["calm and soothe", "hydration"],
        "preferences": {"vegan": True, "fragrance_free": True, "budget_range": [15, 60]},
        "routines": {"am": ["gentle cleanser", "calming toner", "light moisturizer", "SPF"], "pm": ["gentle cleanser", "ceramide cream"]},
        "product_history": {"recommended": [], "liked": [], "disliked": []},
        "category_affinities": ["skincare", "haircare", "makeup"],
    },
    {
        "customer_id": "cust_00005",
        "skin_profile": {"type": "normal", "tone": "tan", "concerns": ["dark circles", "glow"], "sensitivities": []},
        "goals": ["glow", "even complexion"],
        "preferences": {"vegan": False, "fragrance_free": False, "budget_range": [25, 90]},
        "routines": {"am": ["cleanser", "serum", "moisturizer", "SPF"], "pm": ["cleanser", "eye cream", "moisturizer"]},
        "product_history": {"recommended": [], "liked": [], "disliked": []},
        "category_affinities": ["makeup", "skincare", "haircare"],
    },
    {
        "customer_id": "cust_00006",
        "skin_profile": {"type": "oily", "tone": "medium", "concerns": ["acne", "enlarged pores", "hyperpigmentation"], "sensitivities": []},
        "goals": ["acne control", "fading dark spots"],
        "preferences": {"vegan": True, "fragrance_free": True, "budget_range": [20, 70]},
        "routines": {"am": ["BHA cleanser", "niacinamide serum", "lightweight moisturizer", "SPF"], "pm": ["cleanser", "AHA toner", "moisturizer"]},
        "product_history": {"recommended": [], "liked": [], "disliked": []},
        "category_affinities": ["skincare", "makeup", "haircare"],
    },
    {
        "customer_id": "cust_00007",
        "skin_profile": {"type": "dry", "tone": "fair", "concerns": ["frizz", "dry hair"], "sensitivities": ["sulfates"]},
        "goals": ["smooth hair", "hydration"],
        "preferences": {"vegan": True, "fragrance_free": False, "budget_range": [20, 60]},
        "routines": {"am": ["sulfate-free shampoo", "conditioner", "leave-in"], "pm": []},
        "product_history": {"recommended": [], "liked": [], "disliked": []},
        "category_affinities": ["haircare", "skincare", "makeup"],
    },
    {
        "customer_id": "cust_00008",
        "skin_profile": {"type": "combination", "tone": "light", "concerns": ["makeup longevity", "hydration"], "sensitivities": []},
        "goals": ["full glam", "long-wearing makeup"],
        "preferences": {"vegan": False, "fragrance_free": False, "budget_range": [30, 100]},
        "routines": {"am": ["primer", "foundation", "concealer", "setting spray"], "pm": ["makeup remover", "cleanser", "night cream"]},
        "product_history": {"recommended": [], "liked": [], "disliked": []},
        "category_affinities": ["makeup", "skincare", "haircare"],
    },
    {
        "customer_id": "cust_00009",
        "skin_profile": {"type": "normal", "tone": "deep", "concerns": ["curly hair", "frizz"], "sensitivities": ["sulfates", "silicones"]},
        "goals": ["defined curls", "moisture retention"],
        "preferences": {"vegan": True, "fragrance_free": False, "budget_range": [25, 75]},
        "routines": {"am": ["co-wash", "leave-in conditioner", "curl cream"], "pm": []},
        "product_history": {"recommended": [], "liked": [], "disliked": []},
        "category_affinities": ["haircare", "skincare", "makeup"],
    },
    {
        "customer_id": "cust_00010",
        "skin_profile": {"type": "sensitive", "tone": "medium", "concerns": ["anti-aging", "dark circles"], "sensitivities": ["fragrance", "parabens"]},
        "goals": ["anti-aging", "bright eyes"],
        "preferences": {"vegan": True, "fragrance_free": True, "budget_range": [40, 120]},
        "routines": {"am": ["gentle cleanser", "vitamin C", "eye cream", "moisturizer", "SPF"], "pm": ["gentle cleanser", "peptide serum", "eye cream", "moisturizer"]},
        "product_history": {"recommended": [], "liked": [], "disliked": []},
        "category_affinities": ["skincare", "makeup", "haircare"],
    },
]

with psycopg.connect(conn_str) as conn:
    with conn.cursor() as cur:
        # Create tables
        cur.execute(CREATE_TABLES)

        # Seed demo customers (upsert)
        for c in DEMO_CUSTOMERS:
            cur.execute("""
                INSERT INTO customer_memory
                    (customer_id, skin_profile, goals, preferences, routines, product_history, category_affinities)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (customer_id) DO UPDATE SET
                    skin_profile = EXCLUDED.skin_profile,
                    goals = EXCLUDED.goals,
                    preferences = EXCLUDED.preferences,
                    routines = EXCLUDED.routines,
                    updated_at = NOW()
            """, (
                c["customer_id"],
                json.dumps(c["skin_profile"]),
                json.dumps(c["goals"]),
                json.dumps(c["preferences"]),
                json.dumps(c["routines"]),
                json.dumps(c["product_history"]),
                json.dumps(c["category_affinities"]),
            ))
        conn.commit()

print(f"Tables created and {len(DEMO_CUSTOMERS)} demo customers seeded.")
