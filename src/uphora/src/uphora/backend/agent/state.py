# src/uphora/src/uphora/backend/agent/state.py
from typing import Any, Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    customer_id: str
    messages: Annotated[Sequence, add_messages]  # LangChain messages (Human/AI/System)
    intent: str | None                            # "advisor" | "shopper" | "coach"
    memory: dict[str, Any]                        # loaded from Lakebase
    products_found: list[dict]                    # products retrieved this turn

def create_initial_state(customer_id: str, messages: list) -> AgentState:
    return AgentState(
        customer_id=customer_id,
        messages=messages,
        intent=None,
        memory={},
        products_found=[],
    )
