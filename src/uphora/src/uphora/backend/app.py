from contextlib import asynccontextmanager
from .core import create_app
from .router import router
from .db import db_manager

@asynccontextmanager
async def lifespan(app):
    try:
        db_manager.initialize()
        db_manager.start_refresh()
    except Exception as e:
        print(f"[app] Lakebase init skipped (no credentials): {e}")
    yield
    await db_manager.close()

app = create_app(routers=[router], lifespan=lifespan)
