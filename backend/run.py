"""Start the API using HOST and PORT from backend/.env (defaults 127.0.0.1:8731).

    python run.py
"""

import uvicorn

from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
