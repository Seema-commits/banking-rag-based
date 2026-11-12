"""Start the API using HOST, PORT and RELOAD from backend/.env or the environment.

    python run.py
"""

import uvicorn

from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.reload)
