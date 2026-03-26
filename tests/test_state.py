# tests/test_state.py
from langchain_core.messages import HumanMessage, AIMessage
from src.uphora.src.uphora.backend.agent.state import AgentState, create_initial_state

def test_create_initial_state_has_required_keys():
    msgs = [HumanMessage(content="I need a serum")]
    state = create_initial_state(customer_id="cust_00001", messages=msgs)
    assert state["customer_id"] == "cust_00001"
    assert state["intent"] is None
    assert state["memory"] == {}
    assert state["products_found"] == []
    assert len(state["messages"]) == 1

def test_state_messages_accumulate_with_add_messages():
    from langchain_core.messages import HumanMessage
    state = create_initial_state("cust_00001", [HumanMessage(content="hello")])
    assert len(state["messages"]) == 1
