"""Database models for SQLAlchemy."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, String, Text, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class NewsArticle(Base):
    """Model for news articles."""

    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(200))
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_duplicate = Column(Boolean, default=False, nullable=False)
    duplicate_of_id = Column(Integer, ForeignKey("news_articles.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    entities = relationship("Entity", back_populates="article", cascade="all, delete-orphan")
    stock_impacts = relationship(
        "StockImpact", back_populates="article", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_article_timestamp", "timestamp"),
        Index("idx_article_is_duplicate", "is_duplicate"),
    )


class Entity(Base):
    """Model for extracted entities."""

    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("news_articles.id"), nullable=False)
    entity_type = Column(String(50), nullable=False)  # company, sector, regulator, person, event
    entity_value = Column(String(200), nullable=False)
    confidence = Column(Float, default=0.9, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    article = relationship("NewsArticle", back_populates="entities")

    # Indexes
    __table_args__ = (
        Index("idx_entity_type", "entity_type"),
        Index("idx_entity_value", "entity_value"),
        Index("idx_entity_article", "article_id"),
    )


class StockImpact(Base):
    """Model for stock impact mappings."""

    __tablename__ = "stock_impacts"

    id = Column(Integer, primary_key=True, index=True)
    article_id = Column(Integer, ForeignKey("news_articles.id"), nullable=False)
    stock_symbol = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    impact_type = Column(String(50), nullable=False)  # direct, sector, regulatory
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    article = relationship("NewsArticle", back_populates="stock_impacts")

    # Indexes
    __table_args__ = (
        Index("idx_stock_symbol", "stock_symbol"),
        Index("idx_stock_article", "article_id"),
        Index("idx_stock_impact_type", "impact_type"),
    )


class Query(Base):
    """Model for storing query history (optional)."""

    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text, nullable=False)
    results_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Indexes
    __table_args__ = (Index("idx_query_timestamp", "created_at"),)

