# AI-Powered Financial News Intelligence System

An intelligent multi-agent system for processing, deduplicating, and querying financial news articles using LangGraph, OpenAI embeddings, and advanced NLP techniques.

## Features

- **Multi-Agent Architecture**: LangGraph-based workflow with specialized agents for ingestion, deduplication, entity extraction, stock impact analysis, and storage
- **Intelligent Deduplication**: Semantic similarity-based duplicate detection using OpenAI's `text-embedding-3-small` embeddings
- **Entity Extraction**: Named Entity Recognition (NER) using spaCy for extracting companies, sectors, regulators, people, and events
- **Stock Impact Mapping**: Automatic mapping of entities to stock symbols with confidence scores
- **Context-Aware Query System**: Semantic search with entity recognition and hierarchical relationship expansion
- **RESTful API**: FastAPI-based API for easy integration
- **Vector & SQL Storage**: ChromaDB for embeddings and SQLite/PostgreSQL for structured data

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AI-Powered-Financial-News-Intelligence-System
   ```

2. **Install dependencies using uv**:
   ```bash
   uv pip install -e .
   ```

3. **Install spaCy model**:
   ```bash
   python -m spacy download en_core_web_lg
   ```
   (If `en_core_web_lg` is not available, it will fall back to `en_core_web_sm`)

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

5. **Initialize the database**:
   The database will be automatically initialized on first run.

## Usage

### CLI Interface

#### Ingest a news article:
```bash
python demo/cli.py ingest \
  --title "RBI Raises Repo Rate" \
  --content "The Reserve Bank of India has increased the repo rate..." \
  --source "Economic Times"
```

#### Query news articles:
```bash
python demo/cli.py query "What is the impact of RBI rate hike on banking stocks?"
```

#### List recent articles:
```bash
python demo/cli.py list --limit 10
```

#### Show statistics:
```bash
python demo/cli.py stats
```

### API Server

Start the FastAPI server:
```bash
uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`

#### API Endpoints

- `POST /api/news/ingest` - Ingest a news article
- `POST /api/query` - Process a query
- `GET /api/news/{article_id}` - Get article by ID
- `GET /api/stocks/{symbol}/news` - Get news for a stock symbol
- `GET /api/health` - Health check

See interactive API documentation at `http://localhost:8000/docs`

### Scripts

#### Ingest mock news dataset:
```bash
python scripts/ingest_news.py data/mock_news.json
```

#### Run benchmarks:
```bash
python scripts/benchmark.py
```

### Python API

```python
from src.core.graph import get_news_processing_graph, get_query_graph
from src.core.state import create_article_state, create_query_state

# Process a news article
article = {
    "title": "RBI Raises Repo Rate",
    "content": "The Reserve Bank of India has increased...",
    "source": "Economic Times"
}

graph = get_news_processing_graph()
state = create_article_state(article)
result = graph.invoke(state)

# Process a query
query = "What is the impact of RBI rate hike on banking stocks?"
graph = get_query_graph()
state = create_query_state(query)
result = graph.invoke(state)
```

## Project Structure

```
financial-news-intelligence/
├── pyproject.toml          # Project configuration (uv)
├── README.md               # This file
├── ARCHITECTURE.md         # System architecture documentation
├── .env.example            # Environment variables template
├── src/
│   ├── agents/            # Multi-agent implementations
│   ├── core/              # State, config, graph
│   ├── services/          # Embedding, NER, LLM, stock mapping
│   ├── database/          # Vector and SQL database
│   ├── api/               # FastAPI application
│   └── utils/             # Utilities
├── data/                  # Mock data and stock mappings
├── tests/                 # Test suite
├── scripts/               # Utility scripts
└── demo/                  # CLI interface
```

## Testing

Run tests with pytest:
```bash
pytest tests/
```

Run specific test files:
```bash
pytest tests/test_agents.py
pytest tests/test_deduplication.py
pytest tests/test_entity_extraction.py
pytest tests/test_query.py
pytest tests/test_integration.py
```

## Configuration

Key configuration options in `.env`:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: LLM model (default: `gpt-4o-mini`)
- `OPENAI_EMBEDDING_MODEL`: Embedding model (default: `text-embedding-3-small`)
- `DEDUPLICATION_THRESHOLD`: Similarity threshold for duplicates (default: `0.90`)
- `QUERY_SIMILARITY_THRESHOLD`: Similarity threshold for queries (default: `0.75`)

## Performance Metrics

Target metrics:
- Deduplication accuracy: ≥95%
- Entity extraction precision: ≥90%
- Query response time: <2s

Run benchmarks to verify:
```bash
python scripts/benchmark.py
```

## Technical Stack

- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small (1536 dimensions)
- **Vector DB**: ChromaDB
- **SQL DB**: SQLite (development), PostgreSQL (production-ready)
- **NER**: spaCy en_core_web_lg
- **API**: FastAPI
- **Agent Framework**: LangGraph
- **Dependency Management**: uv

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

