# System Architecture

## Overview

The Financial News Intelligence System is built using a multi-agent architecture powered by LangGraph. The system processes financial news articles through a pipeline of specialized agents, extracts entities, maps stock impacts, and provides intelligent query capabilities.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer (FastAPI)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Ingest     │  │    Query     │  │    Health    │          │
│  │   Endpoint   │  │   Endpoint   │  │   Endpoint   │          │
│  └──────┬───────┘  └──────┬───────┘  └───────────────┘          │
└─────────┼─────────────────┼────────────────────────────────────┘
          │                 │
          ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Workflows                          │
│                                                                   │
│  News Processing Graph:                                           │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │Ingestion │→ │Deduplication │→ │Entity Extract│            │
│  └──────────┘  └──────┬───────┘  └──────┬───────┘            │
│                      │                  │                      │
│                      ▼                  ▼                      │
│              ┌──────────────┐  ┌──────────────┐               │
│              │   Skip if    │  │Stock Impact │               │
│              │  Duplicate    │  │  Analysis   │               │
│              └──────┬───────┘  └──────┬───────┘               │
│                     │                 │                        │
│                     └────────┬────────┘                        │
│                              ▼                                  │
│                       ┌──────────────┐                         │
│                       │   Storage    │                         │
│                       └──────────────┘                         │
│                                                                   │
│  Query Processing Graph:                                         │
│  ┌──────────┐                                                    │
│  │  Query   │→ Results                                           │
│  └──────────┘                                                    │
└─────────────────────────────────────────────────────────────────┘
          │                 │
          ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Layer                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Ingestion   │  │Deduplication │  │   Entity     │         │
│  │    Agent     │  │    Agent     │  │ Extraction   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │Stock Impact │  │   Storage    │  │    Query     │         │
│  │    Agent     │  │    Agent     │  │    Agent     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Service Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Embedding   │  │     NER      │  │     LLM      │         │
│  │   Service    │  │   Service    │  │   Service    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                   │
│  ┌──────────────┐                                                │
│  │Stock Mapper  │                                                │
│  │   Service    │                                                │
│  └──────────────┘                                                │
└─────────────────────────────────────────────────────────────────┘
          │                 │
          ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Database Layer                             │
│  ┌──────────────┐              ┌──────────────┐                 │
│  │  Vector DB   │              │   SQL DB     │                 │
│  │  (ChromaDB)  │              │ (PostgreSQL) │                 │
│  │              │              │              │                 │
│  │ - Embeddings │              │ - Articles   │                 │
│  │ - Similarity │              │ - Entities   │                 │
│  │   Search     │              │ - Stock      │                 │
│  │              │              │   Impacts    │                 │
│  └──────────────┘              └──────────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Flow

### News Processing Flow

1. **Ingestion Agent**: Cleans and normalizes incoming news articles
2. **Deduplication Agent**: Generates embeddings and checks for semantic duplicates
3. **Entity Extraction Agent**: Extracts companies, sectors, regulators, people, events
4. **Stock Impact Agent**: Maps entities to stock symbols with confidence scores
5. **Storage Agent**: Stores article, entities, and impacts in both databases

### Query Processing Flow

1. **Query Agent**:
   - Extracts entities from query
   - Determines query type (company, sector, regulator, theme)
   - Performs semantic search in vector DB
   - Filters by entities in SQL DB
   - Expands context (company → sector, sector → companies)
   - Ranks and returns results

## Database Schema

### SQL Database (PostgreSQL)

#### `news_articles`
- `id`: Primary key
- `title`: Article title
- `content`: Article content
- `source`: Article source
- `timestamp`: Publication timestamp
- `is_duplicate`: Boolean flag
- `duplicate_of_id`: Reference to original article

#### `entities`
- `id`: Primary key
- `article_id`: Foreign key to news_articles
- `entity_type`: Type (company, sector, regulator, person, event)
- `entity_value`: Entity name/value
- `confidence`: Confidence score

#### `stock_impacts`
- `id`: Primary key
- `article_id`: Foreign key to news_articles
- `stock_symbol`: Stock symbol (e.g., HDFCBANK)
- `confidence`: Confidence score (0.0-1.0)
- `impact_type`: Type (direct, sector, regulatory)

### Vector Database (ChromaDB)

- **Collection**: `news_articles`
- **Embeddings**: 1536-dimensional vectors (text-embedding-3-small)
- **Metadata**: article_id, title, source, timestamp
- **Distance Metric**: Cosine similarity

## Technical Decisions

### Embeddings
- **Model**: OpenAI `text-embedding-3-small`
- **Dimensions**: 1536
- **Rationale**: High quality, consistent API, good performance for semantic similarity

### Vector Database
- **Choice**: ChromaDB
- **Rationale**: Easy setup, good for RAG applications, persistent storage

### SQL Database
- **Choice**: PostgreSQL with asyncpg driver
- **Rationale**: Production-ready, high concurrency, excellent async support, scalable

### NER
- **Model**: spaCy `en_core_web_lg`
- **Rationale**: Better accuracy for financial terms, fallback to `en_core_web_sm`

### LLM
- **Model**: OpenAI GPT-4o-mini
- **Rationale**: Cost-effective, fast, good performance for entity extraction

### API Framework
- **Choice**: FastAPI
- **Rationale**: Async support, auto-documentation, modern Python

## State Management

The system uses a `TypedDict`-based state (`AgentState`) that flows through the LangGraph workflow:

```python
class AgentState(TypedDict):
    article: Optional[Dict[str, Any]]
    article_id: Optional[int]
    is_duplicate: bool
    duplicate_of_id: Optional[int]
    entities: List[Dict[str, Any]]
    stock_impacts: List[Dict[str, Any]]
    query: Optional[str]
    query_entities: Optional[List[Dict[str, Any]]]
    query_type: Optional[str]
    results: Optional[List[Dict[str, Any]]]
    errors: List[str]
    metadata: Dict[str, Any]
```

## Confidence Scoring

### Stock Impact Confidence

- **Direct Mention**: 1.0 (company explicitly mentioned)
- **Sector Impact**: 0.6-0.8 (based on entity confidence)
- **Regulatory Impact**: 0.5-0.9 (variable based on entity confidence)

### Similarity Thresholds

- **Deduplication**: 0.90 (configurable)
- **Query Search**: 0.75 (configurable)

## Performance Considerations

1. **Embedding Caching**: Embeddings are cached to avoid redundant API calls
2. **Batch Processing**: Embedding service supports batch operations (up to 2048)
3. **Database Indexes**: Key fields are indexed for fast queries
4. **Async Operations**: Database operations are async for better concurrency

## Scalability

- **Horizontal Scaling**: API can be scaled with multiple workers
- **Database**: PostgreSQL supports high concurrency
- **Vector DB**: ChromaDB can be scaled or replaced with cloud solutions (Pinecone, Weaviate)
- **Caching**: Consider Redis for frequently accessed data

## Security

- API keys stored in environment variables
- Input validation using Pydantic schemas
- SQL injection protection via SQLAlchemy ORM
- CORS configuration for API access

## Future Enhancements

1. **Real-time Processing**: WebSocket support for streaming news
2. **Advanced Analytics**: Sentiment analysis, trend detection
3. **Multi-language Support**: Extend NER to other languages
4. **Graph Database**: Use Neo4j for entity relationships
5. **ML Models**: Fine-tuned models for financial entity extraction

