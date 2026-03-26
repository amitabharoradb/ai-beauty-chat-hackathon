import json
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from databricks import sql as dbsql
from fastapi.responses import StreamingResponse
from mlflow.types.responses import ResponsesAgentRequest, ChatContext
from .core import create_router
from .models import CustomerOut, ChatRequest
from .agent.agent import AGENT
from .agent.memory import save_memory_sync
from .db import db_manager

_executor = ThreadPoolExecutor(max_workers=4)

router = create_router()

UC_CATALOG = os.getenv("UC_CATALOG", "amitabh_arora_catalog")
UC_SCHEMA = os.getenv("UC_SCHEMA", "uphora_hackathon")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "9465acf928ae5952")


# Hardcoded demo customers — avoids SQL warehouse cold-start on page load
_DEMO_CUSTOMERS = [
    {"id": "cust_00001", "name": "Ava Chen",        "skin_type": "oily",        "skin_tone": "medium"},
    {"id": "cust_00002", "name": "Sofia Rossi",      "skin_type": "dry",         "skin_tone": "fair"},
    {"id": "cust_00003", "name": "Maya Johnson",     "skin_type": "combination", "skin_tone": "light"},
    {"id": "cust_00004", "name": "Priya Patel",      "skin_type": "sensitive",   "skin_tone": "deep"},
    {"id": "cust_00005", "name": "Zara Williams",    "skin_type": "normal",      "skin_tone": "tan"},
    {"id": "cust_00006", "name": "Leila Hassan",     "skin_type": "oily",        "skin_tone": "medium"},
    {"id": "cust_00007", "name": "Emma Tanaka",      "skin_type": "dry",         "skin_tone": "fair"},
    {"id": "cust_00008", "name": "Camille Dubois",   "skin_type": "combination", "skin_tone": "light"},
    {"id": "cust_00009", "name": "Amara Okafor",     "skin_type": "normal",      "skin_tone": "deep"},
    {"id": "cust_00010", "name": "Isabella Martins", "skin_type": "sensitive",   "skin_tone": "medium"},
]

def _fetch_demo_customers() -> list[dict]:
    return _DEMO_CUSTOMERS


@router.get("/customers", response_model=list[CustomerOut], operation_id="listCustomers")
def list_customers():
    rows = _fetch_demo_customers()
    return [CustomerOut(**r) for r in rows]


@router.post("/chat", operation_id="chat")
async def chat(request: ChatRequest):
    """Minimal: call Databricks SDK serving endpoint directly, return JSON."""
    loop = asyncio.get_running_loop()

    def call_llm():
        try:
            from databricks.sdk import WorkspaceClient
            from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
            w = WorkspaceClient()
            resp = w.serving_endpoints.query(
                name="databricks-claude-sonnet-4-6",
                messages=[
                    ChatMessage(role=ChatMessageRole.SYSTEM, content="You are Ava, Uphora's beauty advisor. Give helpful, warm skincare advice."),
                    ChatMessage(role=ChatMessageRole.USER, content=request.message),
                ],
                max_tokens=512,
            )
            text = resp.choices[0].message.content if resp.choices else ""
            return {"text": text, "products": []}
        except Exception as e:
            import traceback
            return {"text": "", "error": traceback.format_exc()}

    result = await loop.run_in_executor(_executor, call_llm)
    return result
