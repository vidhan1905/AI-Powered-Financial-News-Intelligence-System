"""FastAPI application main file."""

import logging
import re
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import asyncpg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.core.config import settings
from src.database.sql_db import get_sql_db

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def ensure_database_exists():
    """Ensure the database exists, create it if it doesn't."""
    try:
        # Parse the database URL
        # Format: postgresql+asyncpg://user:password@host:port/database
        db_url = settings.sql_database_url
        
        # Remove the +asyncpg part if present
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
        
        # Parse the URL
        parsed = urlparse(db_url)
        
        # Extract components
        user = parsed.username or "postgres"
        password = parsed.password or ""
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432
        database = parsed.path.lstrip("/") if parsed.path else "postgres"
        
        # Connect to PostgreSQL's default database to check if target database exists
        admin_conn = await asyncpg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="postgres"  # Connect to default postgres database
        )
        
        try:
            # Check if database exists
            exists = await admin_conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", database
            )
            
            if not exists:
                logger.info(f"Database '{database}' does not exist. Creating it...")
                # Create database (must be done outside transaction)
                await admin_conn.execute(
                    f'CREATE DATABASE "{database}"'
                )
                logger.info(f"Database '{database}' created successfully")
            else:
                logger.info(f"Database '{database}' already exists")
        finally:
            await admin_conn.close()
            
    except Exception as e:
        # If we can't create the database, log the error but don't fail
        # The connection error will be caught later during init_db
        logger.warning(f"Could not ensure database exists: {e}")
        logger.warning("Will attempt to connect anyway - database may need to be created manually")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("Starting up application...")
    
    # Ensure database exists before initializing
    await ensure_database_exists()
    
    sql_db = get_sql_db()
    await sql_db.init_db()
    logger.info("Database initialized")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    await sql_db.close()
    logger.info("Application shut down")


# Create FastAPI app
app = FastAPI(
    title="Financial News Intelligence System",
    description="AI-Powered Financial News Intelligence System with multi-agent LangGraph architecture",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api", tags=["api"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Financial News Intelligence System API",
        "version": "0.1.0",
        "docs": "/docs",
    }

