import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_db_conn():
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
