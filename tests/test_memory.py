import json
import pytest
from unittest.mock import MagicMock, patch
from src.uphora.src.uphora.backend.agent.memory import load_memory_sync, save_memory_sync

def test_load_memory_sync_returns_dict(mock_db_conn, sample_customer_memory):
    conn, cursor = mock_db_conn
    cursor.fetchone.return_value = (
        sample_customer_memory["skin_profile"],
        sample_customer_memory["goals"],
        sample_customer_memory["preferences"],
        sample_customer_memory["routines"],
        sample_customer_memory["product_history"],
        sample_customer_memory["category_affinities"],
    )
    with patch("src.uphora.src.uphora.backend.agent.memory._get_conn", return_value=conn):
        memory = load_memory_sync("cust_00001")
    assert memory["skin_profile"]["type"] == "oily"
    assert "goals" in memory
    assert "preferences" in memory

def test_load_memory_sync_returns_empty_for_unknown_customer(mock_db_conn):
    conn, cursor = mock_db_conn
    cursor.fetchone.return_value = None
    with patch("src.uphora.src.uphora.backend.agent.memory._get_conn", return_value=conn):
        memory = load_memory_sync("unknown_customer")
    assert memory == {}

def test_save_memory_sync_upserts(mock_db_conn, sample_customer_memory):
    conn, cursor = mock_db_conn
    delta = {"product_history": {"recommended": ["prod_001"], "liked": [], "disliked": []}}
    with patch("src.uphora.src.uphora.backend.agent.memory._get_conn", return_value=conn):
        save_memory_sync("cust_00001", sample_customer_memory, delta)
    assert cursor.execute.called
