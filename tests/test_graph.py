import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.uphora.src.uphora.backend.agent.graph import build_graph
from src.uphora.src.uphora.backend.agent.agent import UphoraBeautyAgent
from mlflow.types.responses import ResponsesAgentRequest, ChatContext

def _mock_llm(text="Great recommendation!"):
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=text)
    return llm

def test_graph_runs_end_to_end(sample_customer_memory, sample_products):
    llm = _mock_llm()
    graph = build_graph(llm)
    state = {
        "customer_id": "cust_00001",
        "messages": [HumanMessage(content="What serum for oily skin?")],
        "intent": None,
        "memory": {},
        "products_found": [],
    }
    with patch("src.uphora.src.uphora.backend.agent.nodes.load_memory_sync", return_value=sample_customer_memory), \
         patch("src.uphora.src.uphora.backend.agent.nodes.load_session_summaries_sync", return_value=[]), \
         patch("src.uphora.src.uphora.backend.agent.nodes.search_products", return_value=sample_products):
        result = graph.invoke(state)

    assert result["intent"] in ("advisor", "shopper", "coach")
    assert len(result["messages"]) >= 2   # HumanMessage + AIMessage
    assert llm.invoke.called

def test_graph_routes_to_shopper_for_browse(sample_customer_memory, sample_products):
    llm = _mock_llm()
    graph = build_graph(llm)
    state = {
        "customer_id": "cust_00001",
        "messages": [HumanMessage(content="Show me all your serums under $60")],
        "intent": None,
        "memory": {},
        "products_found": [],
    }
    with patch("src.uphora.src.uphora.backend.agent.nodes.load_memory_sync", return_value=sample_customer_memory), \
         patch("src.uphora.src.uphora.backend.agent.nodes.load_session_summaries_sync", return_value=[]), \
         patch("src.uphora.src.uphora.backend.agent.nodes.search_products", return_value=sample_products):
        result = graph.invoke(state)

    assert result["intent"] == "shopper"

def test_uphora_agent_predict_returns_response(sample_customer_memory, sample_products):
    with patch("src.uphora.src.uphora.backend.agent.agent.ChatDatabricks") as MockLLM, \
         patch("src.uphora.src.uphora.backend.agent.nodes.load_memory_sync", return_value=sample_customer_memory), \
         patch("src.uphora.src.uphora.backend.agent.nodes.load_session_summaries_sync", return_value=[]), \
         patch("src.uphora.src.uphora.backend.agent.nodes.search_products", return_value=sample_products):
        MockLLM.return_value.invoke.return_value = AIMessage(content="Here is my advice!")
        agent = UphoraBeautyAgent()
        request = ResponsesAgentRequest(
            input=[{"role": "user", "content": "What serum for oily skin?"}],
            context=ChatContext(conversation_id="cust_00001")
        )
        response = agent.predict(request)

    assert len(response.output) > 0
