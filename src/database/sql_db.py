"""SQL database service using SQLAlchemy."""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from sqlalchemy import select, String, func, or_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from src.core.config import settings
from src.database.models import Base, Entity, NewsArticle, Query, StockImpact

logger = logging.getLogger(__name__)


class SQLDatabase:
    """Service for SQL database operations."""

    def __init__(self, database_url: Optional[str] = None):
        """Initialize the SQL database.

        Args:
            database_url: Database URL. If None, uses settings.sql_database_url.
        """
        self.database_url = database_url or settings.sql_database_url
        self.engine = create_async_engine(
            self.database_url,
            echo=False,
            future=True,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,   # Recycle connections after 1 hour
        )
        self.async_session_maker = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self) -> None:
        """Initialize database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Close database connections."""
        await self.engine.dispose()

    @asynccontextmanager
    async def get_session(self):
        """Get an async database session."""
        async with self.async_session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def store_article(
        self,
        title: str,
        content: str,
        source: Optional[str] = None,
        is_duplicate: bool = False,
        duplicate_of_id: Optional[int] = None,
    ) -> int:
        """Store a news article.

        Args:
            title: Article title.
            content: Article content.
            source: Article source.
            is_duplicate: Whether this is a duplicate.
            duplicate_of_id: ID of the original article if duplicate.

        Returns:
            Article ID.
        """
        async with self.get_session() as session:
            article = NewsArticle(
                title=title,
                content=content,
                source=source,
                is_duplicate=is_duplicate,
                duplicate_of_id=duplicate_of_id,
            )
            session.add(article)
            await session.flush()
            article_id = article.id
            await session.commit()
            return article_id

    async def get_article(self, article_id: int) -> Optional[NewsArticle]:
        """Get an article by ID.

        Args:
            article_id: Article ID.

        Returns:
            NewsArticle object or None.
        """
        async with self.get_session() as session:
            result = await session.execute(
                select(NewsArticle)
                .options(selectinload(NewsArticle.entities), selectinload(NewsArticle.stock_impacts))
                .where(NewsArticle.id == article_id)
            )
            return result.scalar_one_or_none()

    async def get_entities(self, article_id: int) -> List[Entity]:
        """Get entities for an article.

        Args:
            article_id: Article ID.

        Returns:
            List of Entity objects.
        """
        async with self.get_session() as session:
            result = await session.execute(
                select(Entity).where(Entity.article_id == article_id)
            )
            return list(result.scalars().all())

    async def store_entities(self, article_id: int, entities: List[dict]) -> None:
        """Store entities for an article.

        Args:
            article_id: Article ID.
            entities: List of entity dictionaries with entity_type, entity_value, confidence.
        """
        async with self.get_session() as session:
            for entity_data in entities:
                entity = Entity(
                    article_id=article_id,
                    entity_type=entity_data["entity_type"],
                    entity_value=entity_data["entity_value"],
                    confidence=entity_data.get("confidence", 0.9),
                )
                session.add(entity)
            await session.commit()

    async def get_stock_impacts(self, article_id: int) -> List[StockImpact]:
        """Get stock impacts for an article.

        Args:
            article_id: Article ID.

        Returns:
            List of StockImpact objects.
        """
        async with self.get_session() as session:
            result = await session.execute(
                select(StockImpact).where(StockImpact.article_id == article_id)
            )
            return list(result.scalars().all())

    async def store_stock_impacts(self, article_id: int, impacts: List[dict]) -> None:
        """Store stock impacts for an article.

        Args:
            article_id: Article ID.
            impacts: List of impact dictionaries with stock_symbol, confidence, impact_type.
        """
        async with self.get_session() as session:
            for impact_data in impacts:
                impact = StockImpact(
                    article_id=article_id,
                    stock_symbol=impact_data["stock_symbol"],
                    confidence=impact_data["confidence"],
                    impact_type=impact_data["impact_type"],
                )
                session.add(impact)
            await session.commit()

    async def get_articles_by_stock(self, stock_symbol: str, limit: int = 50) -> List[NewsArticle]:
        """Get articles that impact a specific stock.

        Args:
            stock_symbol: Stock symbol.
            limit: Maximum number of articles to return.

        Returns:
            List of NewsArticle objects.
        """
        async with self.get_session() as session:
            result = await session.execute(
                select(NewsArticle)
                .join(StockImpact)
                .where(StockImpact.stock_symbol == stock_symbol)
                .where(NewsArticle.is_duplicate == False)
                .order_by(NewsArticle.timestamp.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def get_articles_by_entity(
        self, entity_type: str, entity_value: str, limit: int = 50
    ) -> List[NewsArticle]:
        """Get articles containing a specific entity.
        
        Uses case-insensitive matching and partial matching for better results.

        Args:
            entity_type: Entity type.
            entity_value: Entity value.
            limit: Maximum number of articles to return.

        Returns:
            List of NewsArticle objects.
        """
        async with self.get_session() as session:
            # Normalize entity value for matching (case-insensitive)
            entity_value_lower = entity_value.lower().strip()
            
            # Try exact case-insensitive match first
            result = await session.execute(
                select(NewsArticle)
                .join(Entity)
                .where(Entity.entity_type == entity_type)
                .where(func.lower(Entity.entity_value) == entity_value_lower)
                .where(NewsArticle.is_duplicate == False)
                .order_by(NewsArticle.timestamp.desc())
                .limit(limit)
            )
            articles = list(result.scalars().all())
            
            # If no exact match, try partial matching (entity_value contains query)
            if not articles:
                result = await session.execute(
                    select(NewsArticle)
                    .join(Entity)
                    .where(Entity.entity_type == entity_type)
                    .where(func.lower(Entity.entity_value).contains(entity_value_lower))
                    .where(NewsArticle.is_duplicate == False)
                    .order_by(NewsArticle.timestamp.desc())
                    .limit(limit)
                )
                articles = list(result.scalars().all())
            
            return articles

    async def get_recent_articles(self, limit: int = 50) -> List[NewsArticle]:
        """Get recent non-duplicate articles.

        Args:
            limit: Maximum number of articles to return.

        Returns:
            List of NewsArticle objects.
        """
        async with self.get_session() as session:
            result = await session.execute(
                select(NewsArticle)
                .where(NewsArticle.is_duplicate == False)
                .order_by(NewsArticle.timestamp.desc())
                .limit(limit)
            )
            return list(result.scalars().all())


# Global instance
_sql_db: Optional[SQLDatabase] = None


def get_sql_db() -> SQLDatabase:
    """Get or create the global SQL database instance."""
    global _sql_db
    if _sql_db is None:
        _sql_db = SQLDatabase()
    return _sql_db

