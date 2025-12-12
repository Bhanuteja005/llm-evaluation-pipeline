"""Database persistence for evaluation results."""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings
from src.scoring import EvaluationReport
from src.utils import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class EvaluationRecord(Base):
    """Database model for evaluation results."""

    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Metadata
    chat_id = Column(Integer, nullable=False, index=True)
    turn = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    provider = Column(String(50), nullable=False)

    # Input/Output
    user_message = Column(Text, nullable=False)
    model_response = Column(Text, nullable=False)

    # Scores
    overall_quality_score = Column(Float, nullable=False)
    relevance_score = Column(Float, nullable=False)
    completeness_score = Column(Float, nullable=False)
    hallucination_rate = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False)
    estimated_cost_usd = Column(Float, nullable=False)

    # Flags
    passed_thresholds = Column(Boolean, nullable=False)
    latency_flag = Column(Boolean, nullable=False)
    cost_flag = Column(Boolean, nullable=False)

    # Details (JSON)
    relevance_details = Column(JSON)
    factual_details = Column(JSON)
    latency_cost_details = Column(JSON)
    summary = Column(JSON)

    # Indexes for common queries
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)


class Database:
    """Database interface for persistence."""

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database connection.

        Args:
            database_url: Database URL (uses settings if not provided)
        """
        self.database_url = database_url or settings.database_url
        logger.info(f"Connecting to database: {self.database_url}")

        self.engine = create_engine(self.database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created/verified")

    def save_evaluation(self, report: EvaluationReport) -> int:
        """
        Save evaluation report to database.

        Args:
            report: Evaluation report to save

        Returns:
            ID of saved record
        """
        session = self.SessionLocal()
        try:
            record = EvaluationRecord(
                chat_id=report.chat_id,
                turn=report.turn,
                timestamp=datetime.fromisoformat(report.timestamp.replace("Z", "+00:00")),
                provider=report.provider,
                user_message=report.user_message,
                model_response=report.model_response,
                overall_quality_score=report.overall_quality_score,
                relevance_score=report.relevance.relevance_score,
                completeness_score=report.relevance.completeness_score,
                hallucination_rate=report.factual.hallucination_rate,
                latency_ms=report.latency_cost.latency_ms,
                estimated_cost_usd=report.latency_cost.estimated_cost_usd,
                passed_thresholds=report.passed_thresholds,
                latency_flag=report.latency_cost.latency_flag,
                cost_flag=report.latency_cost.cost_flag,
                relevance_details=report.relevance.to_dict(),
                factual_details=report.factual.to_dict(),
                latency_cost_details=report.latency_cost.to_dict(),
                summary=report.summary,
            )

            session.add(record)
            session.commit()
            record_id = record.id

            logger.info(f"Saved evaluation record with ID: {record_id}")
            return record_id

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving evaluation: {e}")
            raise
        finally:
            session.close()

    def get_evaluation(self, record_id: int) -> Optional[EvaluationRecord]:
        """
        Retrieve evaluation record by ID.

        Args:
            record_id: Record ID

        Returns:
            EvaluationRecord or None
        """
        session = self.SessionLocal()
        try:
            record = session.query(EvaluationRecord).filter_by(id=record_id).first()
            return record
        finally:
            session.close()

    def get_evaluations_by_chat(self, chat_id: int) -> List[EvaluationRecord]:
        """
        Get all evaluations for a specific chat.

        Args:
            chat_id: Chat ID

        Returns:
            List of evaluation records
        """
        session = self.SessionLocal()
        try:
            records = (
                session.query(EvaluationRecord)
                .filter_by(chat_id=chat_id)
                .order_by(EvaluationRecord.turn)
                .all()
            )
            return records
        finally:
            session.close()

    def get_statistics(self) -> dict:
        """
        Get aggregate statistics.

        Returns:
            Dictionary with statistics
        """
        session = self.SessionLocal()
        try:
            total = session.query(EvaluationRecord).count()

            if total == 0:
                return {
                    "total_evaluations": 0,
                    "avg_quality_score": 0,
                    "avg_hallucination_rate": 0,
                    "avg_latency_ms": 0,
                    "pass_rate": 0,
                }

            from sqlalchemy import func

            stats = session.query(
                func.count(EvaluationRecord.id).label("count"),
                func.avg(EvaluationRecord.overall_quality_score).label("avg_quality"),
                func.avg(EvaluationRecord.hallucination_rate).label("avg_hallucination"),
                func.avg(EvaluationRecord.latency_ms).label("avg_latency"),
                func.sum(EvaluationRecord.passed_thresholds.cast(Integer)).label("passed"),
            ).first()

            return {
                "total_evaluations": stats.count,
                "avg_quality_score": float(stats.avg_quality or 0),
                "avg_hallucination_rate": float(stats.avg_hallucination or 0),
                "avg_latency_ms": float(stats.avg_latency or 0),
                "pass_rate": (stats.passed / stats.count * 100) if stats.count > 0 else 0,
            }

        finally:
            session.close()
