"""Script to ingest news articles from mock dataset."""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.graph import get_news_processing_graph
from src.core.state import create_article_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def ingest_news_from_file(file_path: str):
    """Ingest news articles from a JSON file.

    Args:
        file_path: Path to JSON file with news articles.
    """
    # Load news articles
    with open(file_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    logger.info(f"Loaded {len(articles)} articles from {file_path}")

    # Get processing graph
    graph = get_news_processing_graph()

    # Process each article
    processed = 0
    duplicates = 0
    errors = 0

    for i, article in enumerate(articles, 1):
        try:
            logger.info(f"Processing article {i}/{len(articles)}: {article['title'][:50]}...")

            # Create state
            state = create_article_state(article)

            # Process through graph
            final_state = graph.invoke(state)

            # Check results
            if final_state.get("errors"):
                logger.error(f"Errors processing article: {final_state['errors']}")
                errors += 1
            elif final_state.get("is_duplicate"):
                logger.info(f"Article {i} is a duplicate (duplicate of {final_state.get('duplicate_of_id')})")
                duplicates += 1
            else:
                logger.info(f"Article {i} processed successfully (ID: {final_state.get('article_id')})")
                processed += 1

        except Exception as e:
            logger.error(f"Error processing article {i}: {e}")
            errors += 1

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Ingestion Summary:")
    logger.info(f"  Total articles: {len(articles)}")
    logger.info(f"  Processed: {processed}")
    logger.info(f"  Duplicates: {duplicates}")
    logger.info(f"  Errors: {errors}")
    logger.info("=" * 50)


if __name__ == "__main__":
    # Default to data/mock_news.json
    data_file = Path(__file__).parent.parent / "data" / "mock_news.json"

    if len(sys.argv) > 1:
        data_file = Path(sys.argv[1])

    if not data_file.exists():
        logger.error(f"File not found: {data_file}")
        sys.exit(1)

    asyncio.run(ingest_news_from_file(str(data_file)))

