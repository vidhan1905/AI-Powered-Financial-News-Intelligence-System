"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class NewsArticleInput(BaseModel):
    """Schema for news article input."""

    title: str = Field(..., description="Article title", min_length=1)
    content: str = Field(..., description="Article content", min_length=1)
    source: Optional[str] = Field(None, description="Article source")
    timestamp: Optional[datetime] = Field(None, description="Article timestamp")


class EntitySchema(BaseModel):
    """Schema for entity."""

    entity_type: str = Field(..., description="Entity type")
    entity_value: str = Field(..., description="Entity value")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)


class StockImpactSchema(BaseModel):
    """Schema for stock impact."""

    stock_symbol: str = Field(..., description="Stock symbol")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    impact_type: str = Field(..., description="Impact type (direct, sector, regulatory)")


class NewsArticleResponse(BaseModel):
    """Schema for news article response."""

    article_id: int
    title: str
    content: str
    source: Optional[str]
    timestamp: Optional[datetime]
    is_duplicate: bool
    duplicate_of_id: Optional[int]
    entities: List[EntitySchema] = []
    stock_impacts: List[StockImpactSchema] = []


class QueryRequest(BaseModel):
    """Schema for query request."""

    query: str = Field(..., description="User query", min_length=1)


class QueryResult(BaseModel):
    """Schema for query result."""

    article_id: int
    title: str
    content: str
    source: Optional[str]
    timestamp: Optional[datetime]
    similarity_score: float
    entities: List[EntitySchema] = []
    stock_impacts: List[StockImpactSchema] = []


class QueryResponse(BaseModel):
    """Schema for query response."""

    query: str
    query_type: Optional[str]
    results_count: int
    results: List[QueryResult] = []


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str
    version: str
    database_connected: bool
    vector_db_size: int


class ErrorResponse(BaseModel):
    """Schema for error response."""

    error: str
    detail: Optional[str] = None

