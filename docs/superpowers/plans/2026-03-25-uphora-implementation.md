# Uphora AI Beauty Chat — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Uphora, a memory-driven AI beauty chatbot deployed on Databricks Apps using LangGraph, Lakebase Autoscaling, and Unity Catalog.

**Architecture:** LangGraph agent with 5 nodes (memory_loader, intent_router, advisor/shopper/coach, memory_writer) backed by Lakebase Autoscaling (PostgreSQL) for long-term memory and Unity Catalog Delta tables for product/customer data. apx (React + FastAPI) serves the warm-luxury chat UI with customer dropdown and SSE streaming. The graph builds a prompt+context state; FastAPI streams Claude's response directly for real-time token delivery.

**Tech Stack:** Python 3.11+, LangGraph, Databricks Foundation Model API (`databricks-claude-sonnet-4-6` via `databricks-sdk` `w.serving_endpoints.query`), FastAPI, apx (React + Vite), psycopg3, SQLAlchemy async, Databricks SDK ≥0.81, Spark + Faker, Unity Catalog, Lakebase Autoscaling

---

## Environment Variables

Set these before running anything:

```bash
UC_CATALOG=main                  # Unity Catalog catalog name
UC_SCHEMA=uphora                 # Unity Catalog schema name
LAKEBASE_PROJECT_ID=uphora-memory
DATABRICKS_HOST=https://<workspace>.azuredatabricks.net
DATABRICKS_TOKEN=dapi...         # auto-set inside Databricks Apps
DATABRICKS_WAREHOUSE_ID=<sql-warehouse-id>  # find in Databricks SQL > Warehouses
```

---

## File Map

```
ai-beauty-chat-hackathon/
├── notebooks/
│   └── 01_generate_data.py          # Databricks notebook: Spark + Faker
├── scripts/
│   └── setup_lakebase.py            # Create Lakebase project, tables, seed demo customers
├── tests/
│   ├── conftest.py                  # shared fixtures + mocks
│   ├── test_tools.py
│   ├── test_memory.py
│   └── test_nodes.py
└── src/uphora/
    ├── backend/
    │   ├── app.py                   # FastAPI entrypoint (create_app + lifespan)
    │   ├── router.py                # /api/customers, /api/chat (SSE)
    │   ├── models.py                # Pydantic models
    │   ├── db.py                    # Lakebase async connection manager
    │   └── agent/
    │       ├── __init__.py
    │       ├── state.py             # AgentState TypedDict
    │       ├── tools.py             # search_products(), get_routine()
    │       ├── memory.py            # load_memory(), save_memory()
    │       ├── nodes.py             # all 5 LangGraph nodes
    │       └── graph.py             # build_graph() → CompiledGraph
    └── ui/
        ├── routes/index.tsx         # main page (CustomerSelector + ChatWindow)
        ├── components/
        │   ├── CustomerSelector.tsx
        │   ├── ChatWindow.tsx
        │   └── ProductCard.tsx
        ├── hooks/useChat.ts         # SSE streaming hook
        └── styles/uphora.css        # warm luxury theme tokens
```

---

## Phase 1: Bootstrap & Data Layer

### Task 1: Initialize apx project and dependencies

**Files:**
- Create: `src/uphora/` (via `apx init`)
- Create: `tests/conftest.py`
- Modify: `pyproject.toml` (add backend deps)

- [ ] **Step 1: Run apx init**

```bash
cd /path/to/ai-beauty-chat-hackathon
apx init uphora
```

Expected: `src/uphora/backend/` and `src/uphora/ui/` directories created.

- [ ] **Step 2: Add Python dependencies to pyproject.toml**

In `src/uphora/pyproject.toml` (or root `pyproject.toml`), add to `[project.dependencies]`:

```toml
[project]
dependencies = [
  "langgraph>=0.2.0",
  "databricks-sdk>=0.81.0",
  "psycopg[binary]>=3.0",
  "sqlalchemy[asyncio]>=2.0",
  "faker>=24.0.0",
  "pydantic-settings>=2.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-asyncio>=0.23",
  "pytest-mock>=3.12",
  "httpx>=0.27",  # for FastAPI TestClient
]
```

- [ ] **Step 3: Install dependencies**

```bash
uv sync
```

Expected: no errors.

- [ ] **Step 4: Create tests/conftest.py**

```python
import pytest
from unittest.mock import MagicMock, AsyncMock

@pytest.fixture
def mock_anthropic(mocker):
    """Mock Anthropic client — returns a canned response."""
    client = MagicMock()
    message = MagicMock()
    message.content = [MagicMock(text="Here are my recommendations for your skin.")]
    client.messages.create.return_value = message
    return client

@pytest.fixture
def mock_db_conn(mocker):
    """Mock psycopg synchronous connection (psycopg uses sync context managers)."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, cursor

@pytest.fixture
def sample_customer_memory():
    return {
        "customer_id": "cust_001",
        "skin_profile": {
            "type": "oily",
            "tone": "medium",
            "concerns": ["acne", "enlarged pores"],
            "sensitivities": ["fragrance"],
        },
        "goals": ["clear skin", "glow"],
        "preferences": {
            "vegan": True,
            "fragrance_free": True,
            "budget_range": [20, 80],
        },
        "product_history": {
            "recommended": ["prod_001", "prod_005"],
            "liked": ["prod_001"],
            "disliked": [],
        },
        "routines": {
            "am": ["cleanser", "toner", "moisturizer", "SPF"],
            "pm": ["cleanser", "serum", "moisturizer"],
        },
        "category_affinities": ["skincare", "makeup", "haircare"],
        "session_summaries": [],
    }

@pytest.fixture
def sample_products():
    return [
        {
            "id": "prod_001",
            "category": "skincare",
            "name": "Hydra-Gel Cleanser",
            "description": "Lightweight gel cleanser for oily skin",
            "price": 38.0,
            "key_ingredients": ["niacinamide", "salicylic acid"],
            "benefits": ["deep cleanse", "pore minimizing"],
            "tags": ["oily skin", "acne-prone", "vegan"],
        },
        {
            "id": "prod_002",
            "category": "skincare",
            "name": "Glow Serum",
            "description": "Brightening vitamin C serum",
            "price": 54.0,
            "key_ingredients": ["vitamin C", "hyaluronic acid"],
            "benefits": ["brightening", "hydration"],
            "tags": ["all skin types", "vegan", "fragrance-free"],
        },
    ]
```

- [ ] **Step 5: Commit**

```bash
git add src/uphora/ tests/conftest.py pyproject.toml
git commit -m "bootstrap: apx project init + deps + test fixtures"
```

---

### Task 2: Generate Fake Data (Databricks Notebook)

**Files:**
- Create: `notebooks/01_generate_data.py`

- [ ] **Step 1: Write the notebook**

```python
# notebooks/01_generate_data.py
# Databricks notebook source

# COMMAND ----------
# MAGIC %pip install faker==24.0.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import os
import json
import random
from datetime import datetime, timedelta
from faker import Faker
from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()
fake = Faker()
Faker.seed(42)
random.seed(42)

UC_CATALOG = os.getenv("UC_CATALOG", "main")
UC_SCHEMA = os.getenv("UC_SCHEMA", "uphora")

spark.sql(f"CREATE CATALOG IF NOT EXISTS {UC_CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_CATALOG}.{UC_SCHEMA}")

# COMMAND ----------
# Categories
CATEGORIES = [
    {"id": "cat_skincare", "name": "Skincare"},
    {"id": "cat_makeup",   "name": "Makeup"},
    {"id": "cat_haircare", "name": "Haircare"},
]

# Products — 5-15 per category (Sephora-inspired)
PRODUCTS = [
    # Skincare (7)
    {"id":"prod_sk_001","category_id":"cat_skincare","name":"Hydra-Gel Cleanser","description":"Lightweight gel cleanser for oily and combination skin","price":38.0,"key_ingredients":["niacinamide","salicylic acid"],"benefits":["deep cleanse","pore minimizing"],"tags":["oily skin","acne-prone","vegan"]},
    {"id":"prod_sk_002","category_id":"cat_skincare","name":"Glow Vitamin C Serum","description":"Brightening serum with 15% vitamin C","price":54.0,"key_ingredients":["vitamin C","ferulic acid","hyaluronic acid"],"benefits":["brightening","anti-aging","hydration"],"tags":["all skin","vegan","fragrance-free"]},
    {"id":"prod_sk_003","category_id":"cat_skincare","name":"Barrier Repair Moisturizer","description":"Rich cream that restores the skin barrier","price":46.0,"key_ingredients":["ceramides","squalane","peptides"],"benefits":["hydration","barrier repair","calming"],"tags":["dry skin","sensitive skin","fragrance-free"]},
    {"id":"prod_sk_004","category_id":"cat_skincare","name":"Daily Defense SPF 50","description":"Lightweight sunscreen with no white cast","price":34.0,"key_ingredients":["zinc oxide","niacinamide"],"benefits":["UV protection","pore minimizing"],"tags":["all skin","vegan","water-resistant"]},
    {"id":"prod_sk_005","category_id":"cat_skincare","name":"Pore Refining Toner","description":"BHA toner that unclogs pores and balances skin","price":28.0,"key_ingredients":["2% BHA","witch hazel","aloe vera"],"benefits":["pore minimizing","exfoliating","balancing"],"tags":["oily skin","acne-prone","vegan"]},
    {"id":"prod_sk_006","category_id":"cat_skincare","name":"Firming Eye Cream","description":"Peptide-rich cream for dark circles and puffiness","price":62.0,"key_ingredients":["retinol","caffeine","peptides"],"benefits":["anti-aging","brightening","depuffing"],"tags":["all skin","fragrance-free"]},
    {"id":"prod_sk_007","category_id":"cat_skincare","name":"Overnight Recovery Mask","description":"Sleeping mask that repairs skin overnight","price":48.0,"key_ingredients":["bakuchiol","niacinamide","ceramides"],"benefits":["repair","hydration","glow"],"tags":["all skin","vegan","fragrance-free"]},

    # Makeup (7)
    {"id":"prod_mk_001","category_id":"cat_makeup","name":"Skin Tint Foundation SPF 20","description":"Lightweight buildable coverage with SPF","price":44.0,"key_ingredients":["hyaluronic acid","zinc oxide"],"benefits":["natural finish","SPF","hydrating"],"tags":["all skin","vegan","buildable"]},
    {"id":"prod_mk_002","category_id":"cat_makeup","name":"Precision Concealer","description":"Full coverage concealer for dark circles and blemishes","price":32.0,"key_ingredients":["vitamin E","niacinamide"],"benefits":["full coverage","long-wearing","brightening"],"tags":["all skin","vegan"]},
    {"id":"prod_mk_003","category_id":"cat_makeup","name":"Lash Amplify Mascara","description":"Volumizing mascara for bold lashes","price":26.0,"key_ingredients":["argan oil","biotin"],"benefits":["volume","length","conditioning"],"tags":["all lash types","cruelty-free"]},
    {"id":"prod_mk_004","category_id":"cat_makeup","name":"Satin Lip Color","description":"Comfortable satin-finish lipstick in 12 shades","price":24.0,"key_ingredients":["jojoba oil","vitamin E"],"benefits":["hydrating","pigmented","long-wearing"],"tags":["vegan","fragrance-free"]},
    {"id":"prod_mk_005","category_id":"cat_makeup","name":"Cream Blush Stick","description":"Buildable cream blush in 8 natural shades","price":28.0,"key_ingredients":["shea butter","rose extract"],"benefits":["natural flush","blendable","hydrating"],"tags":["all skin","vegan","fragrance-free"]},
    {"id":"prod_mk_006","category_id":"cat_makeup","name":"Liquid Highlighter","description":"Radiant liquid highlighter for glass skin","price":36.0,"key_ingredients":["pearl extract","hyaluronic acid"],"benefits":["radiance","hydration","buildable glow"],"tags":["all skin","vegan"]},
    {"id":"prod_mk_007","category_id":"cat_makeup","name":"Eyeshadow Palette - Nudes","description":"12-shade neutral palette for everyday looks","price":52.0,"key_ingredients":["vitamin E"],"benefits":["versatile","blendable","long-wearing"],"tags":["vegan","cruelty-free"]},

    # Haircare (6)
    {"id":"prod_hc_001","category_id":"cat_haircare","name":"Hydrating Shampoo","description":"Sulfate-free shampoo for dry and damaged hair","price":28.0,"key_ingredients":["argan oil","keratin","biotin"],"benefits":["hydration","repair","shine"],"tags":["dry hair","color-safe","vegan"]},
    {"id":"prod_hc_002","category_id":"cat_haircare","name":"Repair Conditioner","description":"Deep conditioning treatment for smooth, frizz-free hair","price":28.0,"key_ingredients":["shea butter","amino acids"],"benefits":["smoothing","hydration","frizz control"],"tags":["all hair types","vegan","sulfate-free"]},
    {"id":"prod_hc_003","category_id":"cat_haircare","name":"Intense Hair Mask","description":"Weekly treatment mask for intense repair","price":42.0,"key_ingredients":["coconut oil","keratin","biotin"],"benefits":["deep repair","shine","strengthening"],"tags":["damaged hair","color-safe","vegan"]},
    {"id":"prod_hc_004","category_id":"cat_haircare","name":"Scalp Revival Serum","description":"Balancing serum for a healthy scalp","price":38.0,"key_ingredients":["salicylic acid","peppermint","niacinamide"],"benefits":["scalp balance","soothing","growth support"],"tags":["oily scalp","vegan","fragrance-free"]},
    {"id":"prod_hc_005","category_id":"cat_haircare","name":"Curl Defining Cream","description":"Anti-frizz cream that defines and holds curls","price":32.0,"key_ingredients":["flaxseed","shea butter","aloe vera"],"benefits":["curl definition","frizz control","hydration"],"tags":["curly hair","vegan","sulfate-free"]},
    {"id":"prod_hc_006","category_id":"cat_haircare","name":"Dry Shampoo Powder","description":"Invisible dry shampoo that refreshes and volumizes","price":22.0,"key_ingredients":["rice starch","kaolin clay"],"benefits":["volume","oil absorption","refreshing"],"tags":["all hair","vegan","travel-friendly"]},
]

# COMMAND ----------
# Customers (10,000)
SKIN_TYPES = ["oily", "dry", "combination", "normal", "sensitive"]
SKIN_TONES = ["fair", "light", "medium", "tan", "deep"]
CONCERNS_LIST = ["acne", "dryness", "anti-aging", "hyperpigmentation", "redness", "enlarged pores", "dark circles", "frizz"]
INTERACTION_TYPES = ["purchased", "viewed", "liked"]

customers = []
for i in range(10000):
    customer_id = f"cust_{i+1:05d}"
    customers.append({
        "id": customer_id,
        "name": fake.name(),
        "email": fake.email(),
        "age": random.randint(18, 65),
        "skin_type": random.choice(SKIN_TYPES),
        "skin_tone": random.choice(SKIN_TONES),
        "concerns": json.dumps(random.sample(CONCERNS_LIST, k=random.randint(1, 3))),
    })

# COMMAND ----------
# Customer-product interactions (~50K rows)
product_ids = [p["id"] for p in PRODUCTS]
interactions = []
for cust in customers:
    n = random.randint(2, 8)
    for prod_id in random.sample(product_ids, k=min(n, len(product_ids))):
        interactions.append({
            "customer_id": cust["id"],
            "product_id": prod_id,
            "interaction_type": random.choice(INTERACTION_TYPES),
        })

# COMMAND ----------
# Write to Unity Catalog
def write_table(data, schema, table_name, mode="overwrite"):
    df = spark.createDataFrame(data)
    df.write.format("delta").mode(mode).saveAsTable(f"{UC_CATALOG}.{UC_SCHEMA}.{table_name}")
    count = spark.table(f"{UC_CATALOG}.{UC_SCHEMA}.{table_name}").count()
    print(f"  {table_name}: {count} rows")

print("Writing tables to Unity Catalog...")
write_table(CATEGORIES, None, "categories")
write_table(PRODUCTS, None, "products")
write_table(customers, None, "customers")
write_table(interactions, None, "customer_products")
print("Done.")
```

- [ ] **Step 2: Upload and run the notebook in Databricks**

```bash
databricks workspace import notebooks/01_generate_data.py \
  /Workspace/uphora/01_generate_data \
  --language PYTHON --overwrite
```

Then run it via the Databricks UI or:
```bash
databricks jobs create --json '{
  "name": "uphora-generate-data",
  "tasks": [{"task_key": "gen", "notebook_task": {"notebook_path": "/Workspace/uphora/01_generate_data"}}],
  "job_clusters": [{"job_cluster_key": "main", "new_cluster": {"spark_version": "15.4.x-scala2.12", "node_type_id": "Standard_D4ds_v5", "num_workers": 1}}]
}'
```

- [ ] **Step 3: Verify tables exist**

```bash
databricks sql execute \
  --warehouse-id <warehouse-id> \
  --statement "SELECT table_name, COUNT(*) as cnt FROM (
    SELECT 'customers' as table_name, COUNT(*) as cnt FROM main.uphora.customers
    UNION ALL SELECT 'products', COUNT(*) FROM main.uphora.products
    UNION ALL SELECT 'categories', COUNT(*) FROM main.uphora.categories
    UNION ALL SELECT 'customer_products', COUNT(*) FROM main.uphora.customer_products
  ) GROUP BY table_name"
```

Expected: customers=10000, products=27, categories=3, customer_products≈50K

- [ ] **Step 4: Commit**

```bash
git add notebooks/
git commit -m "feat: fake data generation notebook (10K customers, 27 products)"
```

---

### Task 3: Lakebase Setup & Demo Customer Seeding

**Files:**
- Create: `scripts/setup_lakebase.py`

- [ ] **Step 1: Write setup script**

```python
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

PROJECT_ID = os.getenv("LAKEBASE_PROJECT_ID", "uphora-memory")
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
```

- [ ] **Step 2: Run the script**

```bash
python scripts/setup_lakebase.py
```

Expected: `Tables created and 10 demo customers seeded.`

- [ ] **Step 3: Verify**

```python
# Quick verification — run in Python REPL or add to end of script
with psycopg.connect(conn_str) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM customer_memory")
        assert cur.fetchone()[0] == 10, "Expected 10 demo customers"
        cur.execute("SELECT customer_id FROM customer_memory ORDER BY customer_id LIMIT 3")
        print(cur.fetchall())
```

- [ ] **Step 4: Commit**

```bash
git add scripts/
git commit -m "feat: lakebase setup script + 10 demo customers seeded"
```

---

## Phase 2: LangGraph Agent

### Task 4: Agent State

**Files:**
- Create: `src/uphora/backend/agent/__init__.py`
- Create: `src/uphora/backend/agent/state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_state.py
from src.uphora.backend.agent.state import AgentState, create_initial_state

def test_create_initial_state_has_required_keys():
    state = create_initial_state(customer_id="cust_00001", message="I need a serum")
    assert state["customer_id"] == "cust_00001"
    assert state["current_message"] == "I need a serum"
    assert state["intent"] is None
    assert state["memory"] == {}
    assert state["products_found"] == []
    assert state["system_prompt"] == ""
    assert state["claude_messages"] == []
    assert state["session_id"] is not None

def test_state_has_conversation_history_field():
    state = create_initial_state("cust_00001", "hello")
    assert isinstance(state["conversation_history"], list)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_state.py -v
```

Expected: `ImportError` or `ModuleNotFoundError`

- [ ] **Step 3: Implement state**

```python
# src/uphora/backend/agent/__init__.py
# (empty)
```

```python
# src/uphora/backend/agent/state.py
import uuid
from typing import Any
from typing_extensions import TypedDict

class AgentState(TypedDict):
    customer_id: str
    current_message: str
    session_id: str
    intent: str | None                  # "advisor" | "shopper" | "coach"
    memory: dict[str, Any]             # loaded from Lakebase
    conversation_history: list[dict]   # [{"role": "user"|"assistant", "content": str}]
    products_found: list[dict]         # products retrieved this turn
    system_prompt: str                 # built by advisor/shopper/coach node
    claude_messages: list[dict]        # ready to pass to Claude API

def create_initial_state(customer_id: str, message: str) -> AgentState:
    return AgentState(
        customer_id=customer_id,
        current_message=message,
        session_id=str(uuid.uuid4()),
        intent=None,
        memory={},
        conversation_history=[],
        products_found=[],
        system_prompt="",
        claude_messages=[],
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_state.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/uphora/backend/agent/ tests/test_state.py
git commit -m "feat: agent state TypedDict"
```

---

### Task 5: Agent Tools (Product Search + Routine Retrieval)

**Files:**
- Create: `src/uphora/backend/agent/tools.py`
- Create: `tests/test_tools.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools.py
import pytest
from unittest.mock import MagicMock, patch
from src.uphora.backend.agent.tools import search_products, get_routine

def test_search_products_returns_list(sample_products):
    with patch("src.uphora.backend.agent.tools._query_products") as mock_query:
        mock_query.return_value = sample_products
        results = search_products(query="serum for oily skin", category="skincare", filters={})
    assert isinstance(results, list)
    assert len(results) == 2

def test_search_products_filters_by_category(sample_products):
    with patch("src.uphora.backend.agent.tools._query_products") as mock_query:
        mock_query.return_value = [p for p in sample_products if p["category"] == "skincare"]
        results = search_products(query="serum", category="skincare", filters={})
    assert all(p["category"] == "skincare" for p in results)

def test_search_products_filters_vegan(sample_products):
    with patch("src.uphora.backend.agent.tools._query_products") as mock_query:
        mock_query.return_value = sample_products
        results = search_products(query="moisturizer", category=None, filters={"vegan": True})
    # All returned products should have "vegan" tag
    for p in results:
        assert "vegan" in p["tags"]

def test_get_routine_returns_routine_dict(sample_customer_memory):
    routine = get_routine(memory=sample_customer_memory)
    assert "am" in routine
    assert "pm" in routine
    assert isinstance(routine["am"], list)

def test_get_routine_empty_when_no_memory():
    routine = get_routine(memory={})
    assert routine == {"am": [], "pm": []}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tools.py -v
```

Expected: ImportError

- [ ] **Step 3: Implement tools**

```python
# src/uphora/backend/agent/tools.py
import os
from databricks import sql as dbsql

UC_CATALOG = os.getenv("UC_CATALOG", "main")
UC_SCHEMA = os.getenv("UC_SCHEMA", "uphora")
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")

def _query_products(where_clause: str = "1=1", limit: int = 10) -> list[dict]:
    """Query products from Unity Catalog."""
    with dbsql.connect(
        server_hostname=DATABRICKS_HOST.replace("https://", ""),
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        access_token=DATABRICKS_TOKEN,
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
        # map category name to id
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

    # Simple text search on name + description + tags
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tools.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/uphora/backend/agent/tools.py tests/test_tools.py
git commit -m "feat: agent tools (search_products, get_routine)"
```

---

### Task 6: Memory (Lakebase load/save)

**Files:**
- Create: `src/uphora/backend/agent/memory.py`
- Create: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_memory.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.uphora.backend.agent.memory import load_memory, save_memory

@pytest.mark.asyncio
async def test_load_memory_returns_dict(mock_db_conn, sample_customer_memory):
    conn, cursor = mock_db_conn
    cursor.fetchone.return_value = (
        sample_customer_memory["skin_profile"],
        sample_customer_memory["goals"],
        sample_customer_memory["preferences"],
        sample_customer_memory["routines"],
        sample_customer_memory["product_history"],
        sample_customer_memory["category_affinities"],
    )
    with patch("src.uphora.backend.agent.memory._get_conn", return_value=conn):
        memory = await load_memory("cust_00001")
    assert memory["skin_profile"]["type"] == "oily"
    assert "goals" in memory
    assert "preferences" in memory

@pytest.mark.asyncio
async def test_load_memory_returns_empty_for_unknown_customer(mock_db_conn):
    conn, cursor = mock_db_conn
    cursor.fetchone.return_value = None
    with patch("src.uphora.backend.agent.memory._get_conn", return_value=conn):
        memory = await load_memory("unknown_customer")
    assert memory == {}

@pytest.mark.asyncio
async def test_save_memory_upserts(mock_db_conn, sample_customer_memory):
    conn, cursor = mock_db_conn
    delta = {"product_history": {"recommended": ["prod_001"], "liked": [], "disliked": []}}
    with patch("src.uphora.backend.agent.memory._get_conn", return_value=conn):
        await save_memory("cust_00001", sample_customer_memory, delta)
    # Verify execute was called (upsert)
    assert cursor.execute.called
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_memory.py -v
```

Expected: ImportError

- [ ] **Step 3: Implement memory module**

```python
# src/uphora/backend/agent/memory.py
import json
import os
import psycopg
from databricks.sdk import WorkspaceClient

LAKEBASE_PROJECT_ID = os.getenv("LAKEBASE_PROJECT_ID", "uphora-memory")
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


async def load_memory(customer_id: str) -> dict:
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


async def save_memory(customer_id: str, memory: dict, delta: dict) -> None:
    """Upsert memory delta back to Lakebase."""
    # Merge delta into memory
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


async def save_session_summary(customer_id: str, session_id: str, summary: str) -> None:
    """Persist a session summary (keep last 5 per customer)."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO session_summaries (customer_id, session_id, summary)
                VALUES (%s, %s, %s)
            """, (customer_id, session_id, summary))
            # Prune — keep only 5 most recent
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


async def load_session_summaries(customer_id: str) -> list[str]:
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_memory.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/uphora/backend/agent/memory.py tests/test_memory.py
git commit -m "feat: lakebase memory load/save functions"
```

---

### Task 7: LangGraph Nodes

**Files:**
- Create: `src/uphora/backend/agent/nodes.py`
- Create: `tests/test_nodes.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_nodes.py
import pytest
from unittest.mock import AsyncMock, patch
from src.uphora.backend.agent.state import create_initial_state
from src.uphora.backend.agent.nodes import (
    memory_loader_node,
    intent_router_node,
    advisor_node,
    shopper_node,
    coach_node,
)

@pytest.mark.asyncio
async def test_memory_loader_populates_memory(sample_customer_memory):
    state = create_initial_state("cust_00001", "I need help with my skin")
    with patch("src.uphora.backend.agent.nodes.load_memory", AsyncMock(return_value=sample_customer_memory)):
        result = await memory_loader_node(state)
    assert result["memory"]["skin_profile"]["type"] == "oily"

@pytest.mark.asyncio
async def test_intent_router_classifies_skincare_as_advisor():
    state = create_initial_state("cust_00001", "What serum should I use for acne?")
    state["memory"] = {"skin_profile": {"type": "oily"}}
    result = await intent_router_node(state)
    assert result["intent"] in ("advisor", "shopper", "coach")

@pytest.mark.asyncio
async def test_intent_router_classifies_product_search_as_shopper():
    state = create_initial_state("cust_00001", "Show me all your moisturizers under $50")
    state["memory"] = {}
    result = await intent_router_node(state)
    assert result["intent"] == "shopper"

@pytest.mark.asyncio
async def test_advisor_node_sets_system_prompt(sample_customer_memory, sample_products):
    state = create_initial_state("cust_00001", "What should I use for oily skin?")
    state["memory"] = sample_customer_memory
    state["intent"] = "advisor"
    with patch("src.uphora.backend.agent.nodes.search_products", return_value=sample_products):
        result = await advisor_node(state)
    assert len(result["system_prompt"]) > 50
    assert len(result["claude_messages"]) > 0
    assert result["products_found"] == sample_products

@pytest.mark.asyncio
async def test_shopper_node_sets_system_prompt(sample_customer_memory, sample_products):
    state = create_initial_state("cust_00001", "Show me serums")
    state["memory"] = sample_customer_memory
    state["intent"] = "shopper"
    with patch("src.uphora.backend.agent.nodes.search_products", return_value=sample_products):
        result = await shopper_node(state)
    assert len(result["system_prompt"]) > 50
    assert len(result["products_found"]) > 0

@pytest.mark.asyncio
async def test_coach_node_sets_system_prompt(sample_customer_memory):
    state = create_initial_state("cust_00001", "Teach me about double cleansing")
    state["memory"] = sample_customer_memory
    state["intent"] = "coach"
    result = await coach_node(state)
    assert len(result["system_prompt"]) > 50
    assert "coach" in result["system_prompt"].lower() or "routine" in result["system_prompt"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_nodes.py -v
```

Expected: ImportError

- [ ] **Step 3: Implement nodes**

```python
# src/uphora/backend/agent/nodes.py
"""
LangGraph nodes for the Uphora beauty agent.
Note: Nodes build system_prompt and claude_messages but do NOT call Claude.
      Claude is streamed from the FastAPI SSE endpoint after graph.invoke().
"""
import json
from .memory import load_memory, load_session_summaries
from .tools import search_products, get_routine
from .state import AgentState

# ── helpers ────────────────────────────────────────────────────────────────

def _format_skin_profile(memory: dict) -> str:
    sp = memory.get("skin_profile", {})
    if not sp:
        return "No skin profile on file."
    lines = [
        f"Skin type: {sp.get('type', 'unknown')}",
        f"Skin tone: {sp.get('tone', 'unknown')}",
        f"Concerns: {', '.join(sp.get('concerns', [])) or 'none'}",
        f"Sensitivities: {', '.join(sp.get('sensitivities', [])) or 'none'}",
    ]
    prefs = memory.get("preferences", {})
    if prefs:
        lines.append(f"Prefers vegan: {prefs.get('vegan', False)}")
        lines.append(f"Prefers fragrance-free: {prefs.get('fragrance_free', False)}")
        budget = prefs.get("budget_range", [])
        if budget:
            lines.append(f"Budget range: ${budget[0]}–${budget[1]}")
    return "\n".join(lines)


def _format_products_for_prompt(products: list[dict]) -> str:
    if not products:
        return "No products found."
    lines = []
    for p in products:
        lines.append(
            f"- {p['name']} (${p.get('price', '?')}): {p.get('description', '')}. "
            f"Key ingredients: {', '.join(p.get('key_ingredients', []))}."
        )
    return "\n".join(lines)


# ── nodes ──────────────────────────────────────────────────────────────────

async def memory_loader_node(state: AgentState) -> dict:
    """Load customer long-term memory from Lakebase."""
    customer_id = state["customer_id"]
    memory = await load_memory(customer_id)
    summaries = await load_session_summaries(customer_id)
    if summaries:
        memory["session_summaries"] = summaries
    return {"memory": memory}


async def intent_router_node(state: AgentState) -> dict:
    """
    Rule-based intent classification. Fast, no LLM call.
    - "shopper"  → user wants to browse/compare products
    - "coach"    → user wants education, routines, tips
    - "advisor"  → default: personalized recommendation
    """
    msg = state["current_message"].lower()

    shopper_keywords = ["show me", "browse", "find", "search", "compare", "all your", "list", "under $", "price"]
    coach_keywords = ["teach", "how to", "explain", "what is", "routine", "tip", "tutorial", "step by step", "difference between"]

    if any(kw in msg for kw in shopper_keywords):
        return {"intent": "shopper"}
    if any(kw in msg for kw in coach_keywords):
        return {"intent": "coach"}
    return {"intent": "advisor"}


async def advisor_node(state: AgentState) -> dict:
    """Build personalized recommendation prompt based on customer profile."""
    memory = state["memory"]
    message = state["current_message"]
    history = state["conversation_history"]

    # Infer filters from preferences
    prefs = memory.get("preferences", {})
    filters = {
        "vegan": prefs.get("vegan", False),
        "fragrance_free": prefs.get("fragrance_free", False),
    }
    if prefs.get("budget_range"):
        filters["max_price"] = prefs["budget_range"][1]

    # Get top category from affinities
    affinities = memory.get("category_affinities", ["skincare"])
    top_category = affinities[0] if affinities else "skincare"

    products = search_products(query=message, category=top_category, filters=filters)

    skin_summary = _format_skin_profile(memory)
    goals = memory.get("goals", [])
    product_list = _format_products_for_prompt(products)

    system_prompt = f"""You are Ava, Uphora's personal beauty advisor. You give warm, expert, personalized recommendations.

Customer profile:
{skin_summary}
Beauty goals: {', '.join(goals) or 'general beauty'}

Relevant Uphora products:
{product_list}

Guidelines:
- Always reference the customer's specific skin type and concerns.
- Recommend 1-3 products maximum per response. Be specific about why each suits their profile.
- If a product conflicts with their sensitivities, never recommend it.
- End with one follow-up question to deepen personalization.
- Keep tone warm, expert, and concise. No filler phrases."""

    claude_messages = [
        *[{"role": m["role"], "content": m["content"]} for m in history],
        {"role": "user", "content": message},
    ]

    return {
        "system_prompt": system_prompt,
        "claude_messages": claude_messages,
        "products_found": products,
    }


async def shopper_node(state: AgentState) -> dict:
    """Build product discovery prompt."""
    memory = state["memory"]
    message = state["current_message"]
    history = state["conversation_history"]

    prefs = memory.get("preferences", {})
    filters = {
        "vegan": prefs.get("vegan", False),
        "fragrance_free": prefs.get("fragrance_free", False),
    }

    products = search_products(query=message, category=None, filters=filters, limit=6)
    product_list = _format_products_for_prompt(products)

    system_prompt = f"""You are Ava, Uphora's beauty shopping assistant. Help customers discover and compare products.

Available Uphora products matching this request:
{product_list}

Guidelines:
- Present products clearly with price and key benefit.
- Offer to narrow down by skin type, concern, or budget if the list is long.
- Never invent products — only reference those listed above.
- Keep tone helpful and efficient."""

    claude_messages = [
        *[{"role": m["role"], "content": m["content"]} for m in history],
        {"role": "user", "content": message},
    ]

    return {
        "system_prompt": system_prompt,
        "claude_messages": claude_messages,
        "products_found": products,
    }


async def coach_node(state: AgentState) -> dict:
    """Build beauty education and routine coaching prompt."""
    memory = state["memory"]
    message = state["current_message"]
    history = state["conversation_history"]

    skin_summary = _format_skin_profile(memory)
    routine = get_routine(memory)
    am_routine = " → ".join(routine["am"]) if routine["am"] else "not set"
    pm_routine = " → ".join(routine["pm"]) if routine["pm"] else "not set"

    system_prompt = f"""You are Ava, Uphora's beauty coach. You educate customers on skincare routines, techniques, and ingredients.

Customer profile:
{skin_summary}

Their current routines:
AM: {am_routine}
PM: {pm_routine}

Guidelines:
- Explain concepts clearly — assume the customer is learning.
- Connect advice to their specific skin profile when relevant.
- Suggest Uphora products only when directly applicable (do not force it).
- Keep responses focused, warm, and educational.
- Use numbered steps for routines."""

    claude_messages = [
        *[{"role": m["role"], "content": m["content"]} for m in history],
        {"role": "user", "content": message},
    ]

    return {
        "system_prompt": system_prompt,
        "claude_messages": claude_messages,
        "products_found": [],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_nodes.py -v
```

Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/uphora/backend/agent/nodes.py tests/test_nodes.py
git commit -m "feat: langgraph nodes (memory_loader, intent_router, advisor, shopper, coach)"
```

---

### Task 8: LangGraph Graph Assembly

**Files:**
- Create: `src/uphora/backend/agent/graph.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph.py
import pytest
from unittest.mock import AsyncMock, patch
from src.uphora.backend.agent.graph import build_graph
from src.uphora.backend.agent.state import create_initial_state

@pytest.mark.asyncio
async def test_graph_runs_end_to_end(sample_customer_memory, sample_products):
    graph = build_graph()
    state = create_initial_state("cust_00001", "What serum for oily skin?")

    with patch("src.uphora.backend.agent.nodes.load_memory", AsyncMock(return_value=sample_customer_memory)), \
         patch("src.uphora.backend.agent.nodes.load_session_summaries", AsyncMock(return_value=[])), \
         patch("src.uphora.backend.agent.nodes.search_products", return_value=sample_products):
        result = await graph.ainvoke(state)

    assert result["intent"] in ("advisor", "shopper", "coach")
    assert len(result["system_prompt"]) > 0
    assert len(result["claude_messages"]) > 0

@pytest.mark.asyncio
async def test_graph_routes_to_shopper_for_browse(sample_customer_memory, sample_products):
    graph = build_graph()
    state = create_initial_state("cust_00001", "Show me all your serums under $60")

    with patch("src.uphora.backend.agent.nodes.load_memory", AsyncMock(return_value=sample_customer_memory)), \
         patch("src.uphora.backend.agent.nodes.load_session_summaries", AsyncMock(return_value=[])), \
         patch("src.uphora.backend.agent.nodes.search_products", return_value=sample_products):
        result = await graph.ainvoke(state)

    assert result["intent"] == "shopper"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_graph.py -v
```

Expected: ImportError

- [ ] **Step 3: Implement graph**

```python
# src/uphora/backend/agent/graph.py
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    memory_loader_node,
    intent_router_node,
    advisor_node,
    shopper_node,
    coach_node,
)

def _route_intent(state: AgentState) -> str:
    return state["intent"] or "advisor"

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("memory_loader",  memory_loader_node)
    g.add_node("intent_router",  intent_router_node)
    g.add_node("advisor",        advisor_node)
    g.add_node("shopper",        shopper_node)
    g.add_node("coach",          coach_node)

    g.set_entry_point("memory_loader")
    g.add_edge("memory_loader", "intent_router")
    g.add_conditional_edges(
        "intent_router",
        _route_intent,
        {"advisor": "advisor", "shopper": "shopper", "coach": "coach"},
    )
    g.add_edge("advisor", END)
    g.add_edge("shopper", END)
    g.add_edge("coach",   END)

    return g.compile()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_graph.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Run all agent tests**

```bash
pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/uphora/backend/agent/graph.py tests/test_graph.py
git commit -m "feat: langgraph graph assembly + integration tests"
```

---

## Phase 3: FastAPI Backend

### Task 9: Lakebase Connection Manager + Models

**Files:**
- Create: `src/uphora/backend/db.py`
- Create: `src/uphora/backend/models.py`

- [ ] **Step 1: Create db.py**

```python
# src/uphora/backend/db.py
import asyncio
import json
import os
from contextlib import asynccontextmanager
from databricks.sdk import WorkspaceClient

LAKEBASE_PROJECT_ID = os.getenv("LAKEBASE_PROJECT_ID", "uphora-memory")
DB_NAME = "databricks_postgres"

class LakebaseManager:
    """Manages Lakebase connection with token refresh."""

    def __init__(self):
        self._conn_params: dict = {}
        self._refresh_task: asyncio.Task | None = None

    def _fetch_conn_params(self) -> dict:
        w = WorkspaceClient()
        endpoints = list(w.postgres.list_endpoints(
            parent=f"projects/{LAKEBASE_PROJECT_ID}/branches/production"
        ))
        ep_name = endpoints[0].name
        endpoint = w.postgres.get_endpoint(name=ep_name)
        host = endpoint.status.hosts.host
        cred = w.postgres.generate_database_credential(endpoint=ep_name)
        username = w.current_user.me().user_name
        return {"host": host, "dbname": DB_NAME, "user": username, "password": cred.token, "sslmode": "require"}

    def initialize(self) -> None:
        self._conn_params = self._fetch_conn_params()

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(50 * 60)  # refresh at 50 min
            try:
                self._conn_params = await asyncio.to_thread(self._fetch_conn_params)
            except Exception as e:
                print(f"[db] Token refresh failed: {e}")

    def start_refresh(self) -> None:
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def close(self) -> None:
        if self._refresh_task:
            self._refresh_task.cancel()

    @property
    def conn_params(self) -> dict:
        return self._conn_params


# Singleton — initialized in app lifespan
db_manager = LakebaseManager()
```

- [ ] **Step 2: Create models.py**

```python
# src/uphora/backend/models.py
from pydantic import BaseModel

class CustomerOut(BaseModel):
    id: str
    name: str
    skin_type: str
    skin_tone: str

class MessageIn(BaseModel):
    role: str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    customer_id: str
    message: str
    history: list[MessageIn] = []

class ProductOut(BaseModel):
    id: str
    name: str
    description: str
    price: float
    key_ingredients: list[str]
    benefits: list[str]
    tags: list[str]
    category: str
```

- [ ] **Step 3: Commit**

```bash
git add src/uphora/backend/db.py src/uphora/backend/models.py
git commit -m "feat: lakebase connection manager + API models"
```

---

### Task 10: FastAPI Routes (/customers + /chat SSE)

**Files:**
- Create: `src/uphora/backend/router.py`
- Modify: `src/uphora/backend/app.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_router.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    from src.uphora.backend.app import app
    return TestClient(app)

def test_list_customers_returns_10(client):
    with patch("src.uphora.backend.router._fetch_demo_customers") as mock_fetch:
        mock_fetch.return_value = [
            {"id": f"cust_{i:05d}", "name": f"Customer {i}", "skin_type": "oily", "skin_tone": "medium"}
            for i in range(1, 11)
        ]
        resp = client.get("/api/customers")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 10
    assert data[0]["id"] == "cust_00001"

def test_chat_returns_sse_stream(client, sample_customer_memory, sample_products):
    with patch("src.uphora.backend.router.GRAPH") as mock_graph, \
         patch("src.uphora.backend.router.w") as mock_w, \
         patch("src.uphora.backend.router.save_memory", AsyncMock()):

        # Mock graph.ainvoke
        mock_graph.ainvoke = AsyncMock(return_value={
            "system_prompt": "You are Ava...",
            "claude_messages": [{"role": "user", "content": "Hi"}],
            "products_found": sample_products,
            "memory": sample_customer_memory,
            "intent": "advisor",
            "session_id": "test-session",
        })

        # Mock Databricks SDK streaming chunks
        def make_chunk(text):
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = text
            return chunk

        mock_w.serving_endpoints.query.return_value = iter([
            make_chunk("Hello "),
            make_chunk("there!"),
        ])

        resp = client.post("/api/chat", json={
            "customer_id": "cust_00001",
            "message": "What serum for oily skin?",
            "history": [],
        })

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_router.py -v
```

Expected: ImportError or 404

- [ ] **Step 3: Implement router.py**

```python
# src/uphora/backend/router.py
import json
import os
from databricks.sdk import WorkspaceClient
from fastapi import Request
from fastapi.responses import StreamingResponse
from .core import create_router
from .models import CustomerOut, ChatRequest
from .agent.graph import build_graph
from .agent.memory import save_memory
from .db import db_manager

GRAPH = build_graph()

# Databricks Foundation Model API — uses DATABRICKS_HOST + DATABRICKS_TOKEN from env (auto-set in Apps)
w = WorkspaceClient()

router = create_router()

def _fetch_demo_customers() -> list[dict]:
    """Fetch 10 demo customers from Unity Catalog via Databricks SQL."""
    import os
    from databricks import sql as dbsql
    UC_CATALOG = os.getenv("UC_CATALOG", "main")
    UC_SCHEMA = os.getenv("UC_SCHEMA", "uphora")
    with dbsql.connect(
        server_hostname=os.getenv("DATABRICKS_HOST", "").replace("https://", ""),
        http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID', '')}",
        access_token=os.getenv("DATABRICKS_TOKEN", ""),
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, name, skin_type, skin_tone
                FROM {UC_CATALOG}.{UC_SCHEMA}.customers
                WHERE id IN (
                    'cust_00001','cust_00002','cust_00003','cust_00004','cust_00005',
                    'cust_00006','cust_00007','cust_00008','cust_00009','cust_00010'
                )
                ORDER BY id
            """)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


@router.get("/customers", response_model=list[CustomerOut], operation_id="listCustomers")
def list_customers():
    rows = _fetch_demo_customers()
    return [CustomerOut(**r) for r in rows]


@router.post("/chat", operation_id="chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    async def event_stream():
        # 1. Build state and run graph (routing + prompt building)
        from .agent.state import create_initial_state
        state = create_initial_state(request.customer_id, request.message)
        state["conversation_history"] = [
            {"role": m.role, "content": m.content} for m in request.history
        ]
        final_state = await GRAPH.ainvoke(state)

        # 2. Stream Claude response via Databricks Foundation Model API
        full_response = ""
        for chunk in w.serving_endpoints.query(
            name="databricks-claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": final_state["system_prompt"]},
                *final_state["claude_messages"],
            ],
            max_tokens=1024,
            stream=True,
        ):
            if chunk.choices:
                text = chunk.choices[0].delta.content or ""
                if text:
                    full_response += text
                    yield f"data: {json.dumps({'text': text})}\n\n"

        # 3. Yield product cards
        products = final_state.get("products_found", [])
        if products:
            yield f"data: {json.dumps({'products': products[:3]})}\n\n"

        # 4. Save memory delta (fire and forget — don't block stream end)
        recommended_ids = [p["id"] for p in products]
        memory = final_state.get("memory", {})
        existing_recommended = memory.get("product_history", {}).get("recommended", [])
        delta = {
            "product_history": {
                **memory.get("product_history", {}),
                "recommended": list(set(existing_recommended + recommended_ids)),
            }
        }
        await save_memory(request.customer_id, memory, delta)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 4: Update app.py with lifespan**

```python
# src/uphora/backend/app.py
from contextlib import asynccontextmanager
from .core import create_app
from .router import router
from .db import db_manager

@asynccontextmanager
async def lifespan(app):
    db_manager.initialize()
    db_manager.start_refresh()
    yield
    await db_manager.close()

app = create_app(routers=[router], lifespan=lifespan)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_router.py -v
```

Expected: 2 tests PASS

- [ ] **Step 6: Smoke test the API locally**

```bash
apx dev start
curl http://localhost:8000/api/customers
```

Expected: JSON array of 10 customers.

- [ ] **Step 7: Commit**

```bash
git add src/uphora/backend/router.py src/uphora/backend/app.py
git commit -m "feat: fastapi routes (/customers + /chat SSE streaming)"
```

---

## Phase 4: React Frontend

### Task 11: Warm Luxury Theme + CustomerSelector

**Files:**
- Create: `src/uphora/ui/styles/uphora.css`
- Create: `src/uphora/ui/components/CustomerSelector.tsx`

- [ ] **Step 1: Create uphora.css theme**

```css
/* src/uphora/ui/styles/uphora.css */
:root {
  --uphora-cream:   #faf8f5;
  --uphora-gold:    #c9a96e;
  --uphora-gold-lt: #e8d5b0;
  --uphora-charcoal:#2d2d2d;
  --uphora-warm-border: #e8ddd0;
  --uphora-white:   #ffffff;
  --uphora-muted:   #8a8075;
}

body {
  background-color: var(--uphora-cream);
  color: var(--uphora-charcoal);
  font-family: 'Inter', system-ui, sans-serif;
}

.uphora-header {
  background: var(--uphora-cream);
  border-bottom: 1px solid var(--uphora-warm-border);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.uphora-logo {
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: 4px;
  color: var(--uphora-charcoal);
}

.uphora-btn {
  background: var(--uphora-charcoal);
  color: var(--uphora-white);
  border: none;
  padding: 10px 20px;
  font-size: 0.8rem;
  letter-spacing: 1.5px;
  cursor: pointer;
  font-family: inherit;
}

.uphora-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.uphora-btn-gold {
  background: var(--uphora-gold);
  color: var(--uphora-white);
}

.uphora-input {
  border: 1px solid var(--uphora-warm-border);
  background: var(--uphora-white);
  padding: 10px 14px;
  font-size: 0.9rem;
  font-family: inherit;
  color: var(--uphora-charcoal);
  flex: 1;
}

.uphora-input:focus {
  outline: 1px solid var(--uphora-gold);
}

.uphora-select {
  border: 1px solid var(--uphora-warm-border);
  background: var(--uphora-white);
  padding: 10px 14px;
  font-size: 0.9rem;
  font-family: inherit;
  color: var(--uphora-charcoal);
  cursor: pointer;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%232d2d2d' stroke-width='1.5' fill='none'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 12px center;
  padding-right: 36px;
}

.bubble-bot {
  background: var(--uphora-white);
  border: 1px solid var(--uphora-warm-border);
  border-radius: 16px 16px 16px 0;
  padding: 12px 16px;
  max-width: 75%;
  font-size: 0.9rem;
  line-height: 1.6;
}

.bubble-user {
  background: var(--uphora-charcoal);
  color: var(--uphora-white);
  border-radius: 16px 16px 0 16px;
  padding: 12px 16px;
  max-width: 75%;
  font-size: 0.9rem;
  line-height: 1.6;
  margin-left: auto;
}

.uphora-label {
  font-size: 0.65rem;
  letter-spacing: 2px;
  color: var(--uphora-gold);
  text-transform: uppercase;
  margin-bottom: 4px;
}
```

- [ ] **Step 2: Create CustomerSelector.tsx**

```tsx
// src/uphora/ui/components/CustomerSelector.tsx
import { Suspense } from "react";
import { useListCustomersSuspense } from "@/lib/api";
import selector from "@/lib/selector";
import "../styles/uphora.css";

interface Props {
  selectedId: string;
  onSelect: (id: string) => void;
}

function CustomerDropdown({ selectedId, onSelect }: Props) {
  const { data: customers } = useListCustomersSuspense(selector());

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="uphora-label">Browsing as</div>
      <select
        className="uphora-select"
        value={selectedId}
        onChange={(e) => onSelect(e.target.value)}
        style={{ minWidth: 220 }}
      >
        <option value="" disabled>Select customer…</option>
        {customers.map((c) => (
          <option key={c.id} value={c.id}>
            {c.name} — {c.skin_type} skin
          </option>
        ))}
      </select>
    </div>
  );
}

export function CustomerSelector({ selectedId, onSelect }: Props) {
  return (
    <Suspense fallback={<div className="uphora-label">Loading customers…</div>}>
      <CustomerDropdown selectedId={selectedId} onSelect={onSelect} />
    </Suspense>
  );
}
```

- [ ] **Step 3: Verify component renders in dev server**

```bash
apx dev start
```

Open http://localhost:5173 — header with UPHORA logo and customer dropdown should appear.

- [ ] **Step 4: Commit**

```bash
git add src/uphora/ui/styles/uphora.css src/uphora/ui/components/CustomerSelector.tsx
git commit -m "feat: warm luxury theme + customer selector dropdown"
```

---

### Task 12: ProductCard Component

**Files:**
- Create: `src/uphora/ui/components/ProductCard.tsx`

- [ ] **Step 1: Create ProductCard.tsx**

```tsx
// src/uphora/ui/components/ProductCard.tsx
interface Product {
  id: string;
  name: string;
  description: string;
  price: number;
  key_ingredients: string[];
  benefits: string[];
  tags: string[];
}

interface Props {
  product: Product;
}

export function ProductCard({ product }: Props) {
  return (
    <div
      style={{
        border: "1px solid #e8ddd0",
        borderRadius: 4,
        padding: "12px 14px",
        background: "#ffffff",
        width: 180,
        flexShrink: 0,
      }}
    >
      {/* Placeholder image */}
      <div
        style={{
          background: "#f5f0eb",
          height: 80,
          borderRadius: 2,
          marginBottom: 10,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <span style={{ fontSize: "1.5rem" }}>✨</span>
      </div>

      <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "#2d2d2d", marginBottom: 2 }}>
        {product.name}
      </div>
      <div style={{ fontSize: "0.7rem", color: "#c9a96e", marginBottom: 6 }}>
        ${product.price.toFixed(2)}
      </div>
      <div style={{ fontSize: "0.65rem", color: "#8a8075", marginBottom: 8, lineHeight: 1.4 }}>
        {product.description.length > 60
          ? product.description.slice(0, 60) + "…"
          : product.description}
      </div>

      <a
        href="#"
        style={{
          display: "block",
          textAlign: "center",
          fontSize: "0.6rem",
          letterSpacing: "1.5px",
          textTransform: "uppercase",
          color: "#2d2d2d",
          textDecoration: "none",
          border: "1px solid #2d2d2d",
          padding: "5px 0",
        }}
      >
        View on Uphora
      </a>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/uphora/ui/components/ProductCard.tsx
git commit -m "feat: product card component"
```

---

### Task 13: Chat Hook + ChatWindow + Main Page

**Files:**
- Create: `src/uphora/ui/hooks/useChat.ts`
- Create: `src/uphora/ui/components/ChatWindow.tsx`
- Modify: `src/uphora/ui/routes/index.tsx`

- [ ] **Step 1: Create useChat.ts SSE hook**

```ts
// src/uphora/ui/hooks/useChat.ts
import { useCallback, useState } from "react";

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface Product {
  id: string;
  name: string;
  description: string;
  price: number;
  key_ingredients: string[];
  benefits: string[];
  tags: string[];
}

export function useChat(customerId: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [products, setProducts] = useState<Product[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!customerId || isStreaming) return;

      const userMsg: Message = { role: "user", content };
      const updatedHistory = [...messages, userMsg];
      setMessages(updatedHistory);
      setIsStreaming(true);
      setProducts([]);

      // Append empty assistant bubble to stream into
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            customer_id: customerId,
            message: content,
            history: messages.map((m) => ({ role: m.role, content: m.content })),
          }),
        });

        const reader = response.body!.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const text = decoder.decode(value, { stream: true });
          for (const line of text.split("\n")) {
            if (!line.startsWith("data: ")) continue;
            const data = line.slice(6).trim();
            if (data === "[DONE]") break;

            try {
              const parsed = JSON.parse(data);
              if (parsed.text) {
                setMessages((prev) => [
                  ...prev.slice(0, -1),
                  { role: "assistant", content: prev[prev.length - 1].content + parsed.text },
                ]);
              }
              if (parsed.products) {
                setProducts(parsed.products);
              }
            } catch {
              // ignore parse errors on partial chunks
            }
          }
        }
      } finally {
        setIsStreaming(false);
      }
    },
    [customerId, messages, isStreaming]
  );

  const clearChat = useCallback(() => {
    setMessages([]);
    setProducts([]);
  }, []);

  return { messages, products, isStreaming, sendMessage, clearChat };
}
```

- [ ] **Step 2: Create ChatWindow.tsx**

```tsx
// src/uphora/ui/components/ChatWindow.tsx
import { useRef, useEffect, useState, KeyboardEvent } from "react";
import { useChat, Message, Product } from "../hooks/useChat";
import { ProductCard } from "./ProductCard";

interface Props {
  customerId: string;
}

export function ChatWindow({ customerId }: Props) {
  const { messages, products, isStreaming, sendMessage, clearChat } = useChat(customerId);
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed) return;
    setInput("");
    sendMessage(trimmed);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 64px)" }}>
      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: "24px 32px", display: "flex", flexDirection: "column", gap: 16 }}>
        {messages.length === 0 && (
          <div style={{ textAlign: "center", marginTop: 80, color: "#8a8075" }}>
            <div style={{ fontSize: "2rem", marginBottom: 12 }}>✨</div>
            <div className="uphora-label" style={{ display: "block", marginBottom: 8 }}>Uphora Beauty Advisor</div>
            <div style={{ fontSize: "0.9rem" }}>
              {customerId
                ? "Ask me anything about your skincare, makeup, or haircare routine."
                : "Select a customer above to begin."}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} style={{ display: "flex", flexDirection: "column", alignItems: msg.role === "user" ? "flex-end" : "flex-start" }}>
            {msg.role === "assistant" && (
              <span className="uphora-label" style={{ marginBottom: 4, marginLeft: 4 }}>AVIA</span>
            )}
            <div className={msg.role === "user" ? "bubble-user" : "bubble-bot"}>
              {msg.content}
              {isStreaming && i === messages.length - 1 && msg.role === "assistant" && (
                <span style={{ color: "#c9a96e", marginLeft: 4 }}>▌</span>
              )}
            </div>
          </div>
        ))}

        {/* Product cards */}
        {products.length > 0 && (
          <div style={{ display: "flex", gap: 12, overflowX: "auto", paddingBottom: 4 }}>
            {products.map((p) => (
              <ProductCard key={p.id} product={p} />
            ))}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div
        style={{
          borderTop: "1px solid #e8ddd0",
          padding: "16px 32px",
          display: "flex",
          gap: 8,
          background: "#faf8f5",
        }}
      >
        <input
          className="uphora-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={customerId ? "Ask about your skin, products, or routine…" : "Select a customer first"}
          disabled={!customerId || isStreaming}
        />
        <button
          className="uphora-btn uphora-btn-gold"
          onClick={handleSend}
          disabled={!customerId || isStreaming || !input.trim()}
        >
          {isStreaming ? "…" : "SEND"}
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Update main page routes/index.tsx**

```tsx
// src/uphora/ui/routes/index.tsx
import { useState } from "react";
import { CustomerSelector } from "../components/CustomerSelector";
import { ChatWindow } from "../components/ChatWindow";
import "../styles/uphora.css";

export function IndexPage() {
  const [customerId, setCustomerId] = useState("");

  const handleCustomerChange = (id: string) => {
    setCustomerId(id);
  };

  return (
    <div style={{ minHeight: "100vh", background: "#faf8f5" }}>
      {/* Header */}
      <header className="uphora-header">
        <span className="uphora-logo">UPHORA</span>
        <CustomerSelector selectedId={customerId} onSelect={handleCustomerChange} />
      </header>

      {/* Chat */}
      <ChatWindow customerId={customerId} />
    </div>
  );
}
```

- [ ] **Step 4: Start dev server and verify end-to-end**

```bash
apx dev start
```

1. Open http://localhost:5173
2. Select a customer from the dropdown (e.g., "Ava Chen — oily skin")
3. Type: "What serum should I use?"
4. Verify: response streams token by token, product cards appear below response
5. Switch customer, verify chat resets

- [ ] **Step 5: Commit**

```bash
git add src/uphora/ui/hooks/useChat.ts src/uphora/ui/components/ChatWindow.tsx src/uphora/ui/routes/index.tsx
git commit -m "feat: chat window + SSE streaming hook + main page assembly"
```

---

## Phase 5: Deployment

### Task 14: Deploy to Databricks Apps

**Files:**
- Create: `app.yaml` (Databricks App manifest)

- [ ] **Step 1: Build the apx app**

```bash
apx build
```

Expected: `dist/` directory created.

- [ ] **Step 2: Create app.yaml**

```yaml
# app.yaml
command:
  - python
  - -m
  - uvicorn
  - uphora.backend.app:app
  - --host
  - 0.0.0.0
  - --port
  - 8000

env:
  - name: UC_CATALOG
    value: main
  - name: UC_SCHEMA
    value: uphora
  - name: LAKEBASE_PROJECT_ID
    value: uphora-memory
  - name: DATABRICKS_WAREHOUSE_ID
    value: <your-warehouse-id>
# Note: DATABRICKS_TOKEN and DATABRICKS_HOST are injected automatically by Databricks Apps.
# No ANTHROPIC_API_KEY needed — Claude is accessed via Databricks Foundation Model API.
```

- [ ] **Step 3: Deploy**

```bash
databricks apps create uphora --description "Uphora AI Beauty Chatbot"
databricks apps deploy uphora --source-code-path .
```

- [ ] **Step 5: Verify deployment**

```bash
databricks apps get uphora
```

Expected: `state: RUNNING`. Open the app URL and run the end-to-end verification from Task 13 Step 4.

- [ ] **Step 6: Final commit**

```bash
git add app.yaml
git commit -m "feat: databricks app deployment manifest"
git push
```

---

## Verification Checklist

| Check | Command / Action |
|-------|-----------------|
| UC tables exist with correct row counts | `pytest tests/ -v` (all pass) + Databricks SQL query |
| Lakebase has 10 demo customers | `python scripts/setup_lakebase.py` → verify count |
| Agent routes correctly | `pytest tests/test_graph.py -v` |
| `/api/customers` returns 10 customers | `curl http://localhost:8000/api/customers` |
| Chat streams tokens | Open app, select customer, send message |
| Product cards appear | Ask "show me serums" — cards render below response |
| Memory isolation | Switch customer mid-session, verify different profile |
| Deployed app runs | `databricks apps get uphora` → RUNNING |
