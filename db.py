"""SQLite by default; set DATABASE_URL=postgresql+psycopg2://user:pass@host/db for Postgres."""
import datetime as dt
import os
import uuid
from typing import Optional

from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///smear.db"), pool_pre_ping=True)
Session = sessionmaker(engine, expire_on_commit=False)


def _now():
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class Scan(Base):
    __tablename__ = "scans"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    actor: Mapped[str] = mapped_column(String(64))
    task: Mapped[str] = mapped_column(String(32))
    model_version: Mapped[str] = mapped_column(String(64))  
    input_sha256: Mapped[str] = mapped_column(String(64))   
    n_cells: Mapped[int] = mapped_column(Integer)
    flagged_fraction: Mapped[float] = mapped_column(Float)
    result: Mapped[dict] = mapped_column(JSON)


class Audit(Base):
    """Append-only: never update or delete rows here."""
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    actor: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64))
    scan_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)


def init_db():
    Base.metadata.create_all(engine)
