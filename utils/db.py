"""SQLite persistence layer for the Moodle Log Analyzer demo."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    source = Column(String, nullable=True)
    user_id = Column(String, nullable=True)
    uploaded_at = Column(DateTime, nullable=False)
    row_count = Column(Integer, nullable=False)
    column_list = Column(JSON, nullable=False)

    runs = relationship(
        "AnalysisRun",
        back_populates="dataset",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    raw_posts = relationship(
        "RawPost",
        back_populates="dataset",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False)
    config_json = Column(JSON, nullable=True)
    status = Column(String, nullable=True)

    dataset = relationship("Dataset", back_populates="runs")
    artifacts = relationship(
        "RunArtifact",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    events = relationship(
        "RunEvent",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class RawPost(Base):
    __tablename__ = "raw_posts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)

    post_id = Column(Integer, nullable=True)
    discussion = Column(Integer, nullable=True)
    parent = Column(Integer, nullable=True)
    userid = Column(Integer, nullable=True)
    userfullname = Column(String, nullable=True)
    created = Column(DateTime, nullable=True)
    modified = Column(DateTime, nullable=True)
    subject = Column(Text, nullable=True)
    message = Column(Text, nullable=True)
    wordcount = Column(Integer, nullable=True)
    charcount = Column(Integer, nullable=True)
    # GDPR note: raw data is isolated from aggregate outputs in separate tables.
    raw_payload = Column(JSON, nullable=False)

    dataset = relationship("Dataset", back_populates="raw_posts")


class RunArtifact(Base):
    __tablename__ = "run_artifacts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    artifact_type = Column(String, nullable=False)
    payload = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False)

    run = relationship("AnalysisRun", back_populates="artifacts")


class MetricObservation(Base):
    __tablename__ = "metric_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    userid = Column(String, nullable=True)
    userfullname = Column(String, nullable=True)
    metric_name = Column(String, nullable=False)
    # GDPR note: metrics are stored without full message content duplication.
    metric_value = Column(Float, nullable=True)


class RankingObservation(Base):
    __tablename__ = "ranking_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    userid = Column(String, nullable=True)
    userfullname = Column(String, nullable=True)
    rank_name = Column(String, nullable=False)
    rank_value = Column(Float, nullable=True)


class RunEvent(Base):
    __tablename__ = "run_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False)
    event_name = Column(String, nullable=False)
    details = Column(Text, nullable=True)
    payload = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False)

    run = relationship("AnalysisRun", back_populates="events")


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _resolve_sqlite_path() -> Path:
    configured = None
    try:
        db_cfg = st.secrets.get("database", {})
        configured = db_cfg.get("SQLITE_PATH")
    except Exception:
        # Streamlit raises when no secrets file exists; fallback to env/default path.
        configured = None
    configured = configured or os.environ.get("APP_SQLITE_PATH")
    if configured:
        db_path = Path(configured).expanduser().resolve()
    else:
        base_dir = Path(__file__).resolve().parents[1]
        db_path = base_dir / "data" / "runs.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def get_database_url() -> str:
    db_path = _resolve_sqlite_path()
    normalized = str(db_path).replace("\\", "/")
    return f"sqlite:///{normalized}"


class DatabaseManager:
    """Thin SQLite persistence layer used for thesis run artifacts."""

    def __init__(self) -> None:
        self.database_url = get_database_url()
        self.enabled = True
        self.engine = create_engine(
            self.database_url,
            future=True,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False},
        )
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        Base.metadata.create_all(self.engine)

    def is_available(self) -> bool:
        return bool(self.enabled and self.engine and self.Session)

    def _session(self):
        return self.Session()

    def create_dataset_and_run(
        self,
        df: pd.DataFrame,
        source_info: Optional[str],
        user_id: Optional[str],
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        dataset_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        now = datetime.utcnow()

        dataset = Dataset(
            id=dataset_id,
            name=source_info,
            source=source_info,
            user_id=user_id,
            uploaded_at=now,
            row_count=int(len(df)),
            column_list=list(df.columns),
        )
        run = AnalysisRun(
            id=run_id,
            dataset_id=dataset_id,
            user_id=user_id,
            created_at=now,
            config_json=config_snapshot,
            status="uploaded",
        )

        session = self._session()
        try:
            session.add(dataset)
            session.add(run)
            session.flush()
            self._persist_raw_posts(session, dataset_id, df)
            session.commit()
        finally:
            session.close()
        return run_id, dataset_id

    def update_run_status(self, run_id: str, status: str) -> None:
        if not self.is_available():
            return
        session = self._session()
        try:
            run = session.query(AnalysisRun).filter(AnalysisRun.id == run_id).one_or_none()
            if run is None:
                return
            run.status = status
            session.commit()
        finally:
            session.close()

    def save_run_event(
        self,
        run_id: str,
        event_name: str,
        details: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.is_available():
            return
        session = self._session()
        try:
            session.add(
                RunEvent(
                    run_id=run_id,
                    event_name=event_name,
                    details=details,
                    payload=json.loads(json.dumps(payload or {}, default=str)),
                    created_at=datetime.utcnow(),
                )
            )
            session.commit()
        finally:
            session.close()

    def _persist_raw_posts(self, session, dataset_id: str, df: pd.DataFrame) -> None:
        records = []
        for row in df.to_dict(orient="records"):
            payload = {key: _safe_json_value(value) for key, value in row.items()}
            records.append(
                {
                    "dataset_id": dataset_id,
                    "post_id": row.get("id"),
                    "discussion": row.get("discussion"),
                    "parent": row.get("parent"),
                    "userid": row.get("userid"),
                    "userfullname": row.get("userfullname"),
                    "created": row.get("created"),
                    "modified": row.get("modified"),
                    "subject": row.get("subject"),
                    "message": row.get("message"),
                    "wordcount": row.get("wordcount"),
                    "charcount": row.get("charcount"),
                    "raw_payload": payload,
                }
            )
        if records:
            session.bulk_insert_mappings(RawPost, records)

    def list_datasets(self, user_id: Optional[str]) -> List[Dict[str, Any]]:
        if not self.is_available():
            return []
        session = self._session()
        try:
            query = session.query(Dataset)
            if user_id:
                query = query.filter(Dataset.user_id == user_id)
            datasets = query.order_by(Dataset.uploaded_at.desc()).all()
            return [
                {
                    "id": row.id,
                    "name": row.name or "Unnamed dataset",
                    "uploaded_at": row.uploaded_at,
                    "row_count": row.row_count,
                }
                for row in datasets
            ]
        finally:
            session.close()

    def delete_dataset(self, dataset_id: str, user_id: Optional[str]) -> bool:
        if not self.is_available():
            return False
        session = self._session()
        try:
            query = session.query(Dataset).filter(Dataset.id == dataset_id)
            if user_id:
                query = query.filter(Dataset.user_id == user_id)
            dataset = query.one_or_none()
            if dataset is None:
                return False
            # GDPR note: cascade delete removes raw rows and all derived outputs.
            session.delete(dataset)
            session.commit()
            return True
        finally:
            session.close()

    def load_raw_data(self, dataset_id: str) -> Optional[pd.DataFrame]:
        if not self.is_available():
            return None
        session = self._session()
        try:
            rows = (
                session.query(RawPost.raw_payload)
                .filter(RawPost.dataset_id == dataset_id)
                .order_by(RawPost.id.asc())
                .all()
            )
            if not rows:
                return None
            records = [row.raw_payload for row in rows]
            df = pd.DataFrame(records)
            for col in ["created", "modified"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            return df
        finally:
            session.close()

    def save_artifact(self, run_id: str, artifact_type: str, payload: Any) -> None:
        if not self.is_available():
            return
        session = self._session()
        try:
            existing = (
                session.query(RunArtifact)
                .filter(RunArtifact.run_id == run_id, RunArtifact.artifact_type == artifact_type)
                .one_or_none()
            )
            serializable = json.loads(json.dumps(payload, default=str))
            if existing:
                existing.payload = serializable
                existing.created_at = datetime.utcnow()
            else:
                session.add(
                    RunArtifact(
                        run_id=run_id,
                        artifact_type=artifact_type,
                        payload=serializable,
                        created_at=datetime.utcnow(),
                    )
                )
            session.commit()
        finally:
            session.close()

    def load_artifact(self, run_id: str, artifact_type: str) -> Optional[Any]:
        if not self.is_available():
            return None
        session = self._session()
        try:
            artifact = (
                session.query(RunArtifact)
                .filter(RunArtifact.run_id == run_id, RunArtifact.artifact_type == artifact_type)
                .one_or_none()
            )
            return artifact.payload if artifact else None
        finally:
            session.close()

    def save_metric_observations(self, run_id: str, df: pd.DataFrame) -> None:
        if not self.is_available():
            return
        metric_rows = []
        for _, row in df.iterrows():
            userid = row.get("userid")
            name = row.get("userfullname")
            for col, value in row.items():
                if col in {"userid", "userfullname"}:
                    continue
                numeric = pd.to_numeric(value, errors="coerce")
                metric_rows.append(
                    {
                        "run_id": run_id,
                        "userid": str(userid) if userid is not None else None,
                        "userfullname": name,
                        "metric_name": col,
                        "metric_value": float(numeric) if pd.notna(numeric) else None,
                    }
                )
        if not metric_rows:
            return
        session = self._session()
        try:
            session.query(MetricObservation).filter(MetricObservation.run_id == run_id).delete()
            session.bulk_insert_mappings(MetricObservation, metric_rows)
            session.commit()
        finally:
            session.close()

    def save_ranking_observations(self, run_id: str, df: pd.DataFrame) -> None:
        if not self.is_available():
            return
        ranking_rows = []
        rank_cols = [col for col in df.columns if col.endswith("_rank")]
        for _, row in df.iterrows():
            userid = row.get("userid")
            name = row.get("userfullname")
            for col in rank_cols:
                numeric = pd.to_numeric(row.get(col), errors="coerce")
                ranking_rows.append(
                    {
                        "run_id": run_id,
                        "userid": str(userid) if userid is not None else None,
                        "userfullname": name,
                        "rank_name": col,
                        "rank_value": float(numeric) if pd.notna(numeric) else None,
                    }
                )
        if not ranking_rows:
            return
        session = self._session()
        try:
            session.query(RankingObservation).filter(RankingObservation.run_id == run_id).delete()
            session.bulk_insert_mappings(RankingObservation, ranking_rows)
            session.commit()
        finally:
            session.close()
