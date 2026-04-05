from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..core.config import settings
from .base import Base

if settings.database_url.startswith("sqlite"):
    _p = settings.database_url.replace("sqlite:///", "")
    Path(_p).parent.mkdir(parents=True, exist_ok=True)

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    import time
    from sqlalchemy.exc import OperationalError

    for i in range(5):
        try:
            Base.metadata.create_all(bind=engine)
            break
        except OperationalError as e:
            if "already exists" in str(e) or "database is locked" in str(e):
                time.sleep(0.5)
                if i == 4:
                    pass  # If it fails on the last try, let the other workers handle it or crash organically later
            else:
                raise
    try:
        from .migrate import backfill_branches, ensure_branch_extra_column

        ensure_branch_extra_column()
        backfill_branches()
    except Exception:
        pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
