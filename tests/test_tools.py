import pytest
from unittest.mock import MagicMock, patch
from src.uphora.src.uphora.backend.agent.tools import search_products, get_routine

def test_search_products_returns_list(sample_products):
    with patch("src.uphora.src.uphora.backend.agent.tools._query_products") as mock_query:
        mock_query.return_value = sample_products
        results = search_products(query="serum for oily skin", category="skincare", filters={})
    assert isinstance(results, list)
    assert len(results) == 2

def test_search_products_filters_by_category(sample_products):
    with patch("src.uphora.src.uphora.backend.agent.tools._query_products") as mock_query:
        mock_query.return_value = [p for p in sample_products if p["category"] == "skincare"]
        results = search_products(query="serum", category="skincare", filters={})
    assert all(p["category"] == "skincare" for p in results)

def test_search_products_filters_vegan(sample_products):
    with patch("src.uphora.src.uphora.backend.agent.tools._query_products") as mock_query:
        mock_query.return_value = sample_products
        results = search_products(query="moisturizer", category=None, filters={"vegan": True})
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
