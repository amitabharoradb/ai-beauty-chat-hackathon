"""
Mosaic AI Agent Framework entry point.
UphoraBeautyAgent wraps the LangGraph graph in a ResponsesAgent for standardized
input/output, MLflow tracing, and future model serving compatibility.
Deployed on Databricks Apps (not a Model Serving endpoint).
"""
import os
import logging
import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from databricks_langchain import ChatDatabricks
from .graph import build_graph
from .memory import save_memory_sync

logger = logging.getLogger(__name__)

LLM_ENDPOINT = "databricks-claude-sonnet-4-6"


class UphoraBeautyAgent(ResponsesAgent):
    def __init__(self):
        self.llm = ChatDatabricks(endpoint=LLM_ENDPOINT, max_tokens=1024)
        self._graph = build_graph(self.llm)
        self._last_products: list[dict] = []

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs)

    def predict_stream(self, request: ResponsesAgentRequest):
        customer_id = (request.context.conversation_id
                       if request.context else "demo")

        messages = to_chat_completions_input(
            [m.model_dump() for m in request.input]
        )

        state = {
            "customer_id": customer_id,
            "messages": messages,
            "intent": None,
            "memory": {},
            "products_found": [],
        }

        self._last_products = []

        # Use invoke (simpler and version-agnostic) instead of stream
        result = self._graph.invoke(state)

        # Extract products
        if result.get("products_found"):
            self._last_products = result["products_found"]

        # Extract the last AI message and yield it
        messages_out = result.get("messages", [])
        ai_messages = [m for m in messages_out if hasattr(m, "content") and not hasattr(m, "type") or (hasattr(m, "type") and m.type == "ai")]
        # Fallback: just get the last message that isn't the user input
        if not ai_messages:
            ai_messages = [m for m in messages_out if hasattr(m, "content")]

        if ai_messages:
            last_msg = ai_messages[-1]
            text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            yield from output_to_responses_items_stream([last_msg])

    def get_last_products(self) -> list[dict]:
        """Return products found in the most recent predict_stream call."""
        return self._last_products


# MLflow 3.0 setup — autolog traces every LangGraph + ChatDatabricks call
mlflow.langchain.autolog()
try:
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "uphora-beauty-chat"))
except Exception as e:
    logger.warning("mlflow.set_experiment failed (no tracking server): %s", e)

try:
    AGENT = UphoraBeautyAgent()
    mlflow.models.set_model(AGENT)
except Exception as e:
    logger.warning("UphoraBeautyAgent module-level init failed: %s", e)
    AGENT = None
