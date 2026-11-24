from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("APP_ENV", "dev")
os.environ.setdefault("LLM_PROVIDER", "fallback")
os.environ.setdefault("ML_ENABLED", "false")
os.environ.setdefault("CACHE_ENABLED", "false")
os.environ.setdefault("QUEUE_ENABLED", "false")


@pytest.fixture(scope="session")
def test_db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    db_file = tmp_path_factory.mktemp("db") / "test_ckgai.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_file.as_posix()}"
    return db_file


@pytest.fixture(scope="session")
def client(test_db_path: Path) -> TestClient:
    from app.core.config import settings
    from app.db.base import Base
    from app.db.session import dispose_engines, get_engine
    import app.models  # noqa: F401
    from app.main import create_app
    from scripts.seed import seed

    settings.database_url = f"sqlite:///{test_db_path.as_posix()}"
    settings.database_url_readonly = ""
    dispose_engines()
    engine = get_engine()
    Base.metadata.create_all(engine)
    seed()

    app = create_app()
    with TestClient(app) as tc:
        yield tc
