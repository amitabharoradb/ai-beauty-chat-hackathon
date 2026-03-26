import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.uphora.src.uphora.backend.agent.state import create_initial_state
from src.uphora.src.uphora.backend.agent.nodes import (
    make_memory_loader_node,
    make_intent_router_node,
    make_advisor_node,
    make_shopper_node,
    make_coach_node,
)

def _make_llm(response_text="Here are my recommendations."):
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=response_text)
    return llm

def test_memory_loader_populates_memory(sample_customer_memory):
    state = create_initial_state("cust_00001", [HumanMessage(content="I need help with my skin")])
    node = make_memory_loader_node()
    with patch("src.uphora.src.uphora.backend.agent.nodes.load_memory_sync", return_value=sample_customer_memory), \
         patch("src.uphora.src.uphora.backend.agent.nodes.load_session_summaries_sync", return_value=[]):
        result = node(state)
    assert result["memory"]["skin_profile"]["type"] == "oily"

def test_intent_router_classifies_product_search_as_shopper():
    state = create_initial_state("cust_00001", [HumanMessage(content="Show me all moisturizers under $50")])
    node = make_intent_router_node()
    result = node(state)
    assert result["intent"] == "shopper"

def test_intent_router_classifies_routine_question_as_coach():
    state = create_initial_state("cust_00001", [HumanMessage(content="How to build a routine for dry skin?")])
    node = make_intent_router_node()
    result = node(state)
    assert result["intent"] == "coach"

def test_intent_router_defaults_to_advisor():
    state = create_initial_state("cust_00001", [HumanMessage(content="My skin is oily and I break out")])
    node = make_intent_router_node()
    result = node(state)
    assert result["intent"] == "advisor"

def test_advisor_node_calls_llm_and_returns_products(sample_customer_memory, sample_products):
    llm = _make_llm("I recommend the Hydra-Gel Cleanser for your oily skin.")
    state = create_initial_state("cust_00001", [HumanMessage(content="What should I use for oily skin?")])
    state["memory"] = sample_customer_memory
    node = make_advisor_node(llm)
    with patch("src.uphora.src.uphora.backend.agent.nodes.search_products", return_value=sample_products):
        result = node(state)
    assert llm.invoke.called
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["products_found"] == sample_products

def test_shopper_node_calls_llm_and_returns_products(sample_customer_memory, sample_products):
    llm = _make_llm("Here are our serums:")
    state = create_initial_state("cust_00001", [HumanMessage(content="Show me serums")])
    state["memory"] = sample_customer_memory
    node = make_shopper_node(llm)
    with patch("src.uphora.src.uphora.backend.agent.nodes.search_products", return_value=sample_products):
        result = node(state)
    assert llm.invoke.called
    assert result["products_found"] == sample_products

def test_coach_node_calls_llm_returns_no_products(sample_customer_memory):
    llm = _make_llm("Double cleansing means using an oil-based then water-based cleanser.")
    state = create_initial_state("cust_00001", [HumanMessage(content="Teach me about double cleansing")])
    state["memory"] = sample_customer_memory
    node = make_coach_node(llm)
    result = node(state)
    assert llm.invoke.called
    assert result["products_found"] == []
