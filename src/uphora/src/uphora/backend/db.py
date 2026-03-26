import asyncio
import os
from databricks.sdk import WorkspaceClient

LAKEBASE_PROJECT_ID = os.getenv("LAKEBASE_PROJECT_ID", "uphora-hackathon-memory")
DB_NAME = "databricks_postgres"

class LakebaseManager:
    """Manages Lakebase connection with token refresh."""

    def __init__(self):
        self._conn_params: dict = {}
        self._refresh_task: asyncio.Task | None = None

    def _fetch_conn_params(self) -> dict:
        w = WorkspaceClient()
        endpoints = list(w.postgres.list_endpoints(
            parent=f"projects/{LAKEBASE_PROJECT_ID}/branches/production"
        ))
        ep_name = endpoints[0].name
        endpoint = w.postgres.get_endpoint(name=ep_name)
        host = endpoint.status.hosts.host
        cred = w.postgres.generate_database_credential(endpoint=ep_name)
        username = w.current_user.me().user_name
        return {"host": host, "dbname": DB_NAME, "user": username, "password": cred.token, "sslmode": "require"}

    def initialize(self) -> None:
        self._conn_params = self._fetch_conn_params()

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(50 * 60)  # refresh at 50 min
            try:
                self._conn_params = await asyncio.to_thread(self._fetch_conn_params)
            except Exception as e:
                print(f"[db] Token refresh failed: {e}")

    def start_refresh(self) -> None:
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def close(self) -> None:
        if self._refresh_task:
            self._refresh_task.cancel()

    @property
    def conn_params(self) -> dict:
        return self._conn_params


# Singleton — initialized in app lifespan
db_manager = LakebaseManager()
