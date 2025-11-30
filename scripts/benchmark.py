"""Benchmark script for measuring system performance."""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.graph import get_news_processing_graph, get_query_graph
from src.core.state import create_article_state, create_query_state
from src.database.sql_db import get_sql_db
from src.database.vector_db import get_vector_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def benchmark_deduplication():
    """Benchmark deduplication accuracy."""
    logger.info("Benchmarking deduplication accuracy...")

    # Load test articles
    data_file = Path(__file__).parent.parent / "data" / "mock_news.json"
    with open(data_file, "r", encoding="utf-8") as f:
        articles = json.load(f)

    graph = get_news_processing_graph()

    # Process all articles
    processed_articles = []
    for article in articles:
        state = create_article_state(article)
        final_state = graph.invoke(state)
        processed_articles.append(final_state)

    # Count duplicates
    duplicates = sum(1 for s in processed_articles if s.get("is_duplicate", False))
    total = len(processed_articles)

    # Expected: Article 4 is duplicate of Article 1
    accuracy = (total - duplicates) / total * 100 if total > 0 else 0

    logger.info(f"Deduplication Results:")
    logger.info(f"  Total articles: {total}")
    logger.info(f"  Duplicates detected: {duplicates}")
    logger.info(f"  Unique articles: {total - duplicates}")
    logger.info(f"  Accuracy: {accuracy:.2f}%")

    return accuracy >= 95.0  # Target: ≥95%


async def benchmark_entity_extraction():
    """Benchmark entity extraction precision."""
    logger.info("Benchmarking entity extraction precision...")

    # Test article with known entities
    test_article = {
        "title": "RBI Raises Repo Rate, Impacting HDFC Bank and ICICI Bank",
        "content": "The Reserve Bank of India (RBI) has increased the repo rate. This affects the banking sector, particularly HDFC Bank and ICICI Bank. The regulator's decision impacts financial services.",
        "source": "Test",
    }

    graph = get_news_processing_graph()
    state = create_article_state(test_article)
    final_state = graph.invoke(state)

    entities = final_state.get("entities", [])

    # Expected entities
    expected_entities = {
        "companies": ["HDFC Bank", "ICICI Bank"],
        "regulators": ["RBI", "Reserve Bank of India"],
        "sectors": ["Banking"],
    }

    # Count matches
    found_entities = {"companies": [], "regulators": [], "sectors": []}
    for entity in entities:
        entity_type = entity.get("entity_type", "")
        entity_value = entity.get("entity_value", "")
        if entity_type in found_entities:
            found_entities[entity_type].append(entity_value)

    # Calculate precision
    total_expected = sum(len(v) for v in expected_entities.values())
    total_found = sum(len(v) for v in found_entities.values())
    matches = 0

    for entity_type, expected_list in expected_entities.items():
        found_list = found_entities.get(entity_type, [])
        for expected in expected_list:
            # Check for partial matches
            if any(expected.lower() in found.lower() or found.lower() in expected.lower() for found in found_list):
                matches += 1

    precision = (matches / total_expected * 100) if total_expected > 0 else 0

    logger.info(f"Entity Extraction Results:")
    logger.info(f"  Expected entities: {total_expected}")
    logger.info(f"  Found entities: {total_found}")
    logger.info(f"  Matches: {matches}")
    logger.info(f"  Precision: {precision:.2f}%")

    return precision >= 90.0  # Target: ≥90%


async def benchmark_query_performance():
    """Benchmark query response time."""
    logger.info("Benchmarking query performance...")

    queries = [
        "What is the impact of RBI rate hike on banking stocks?",
        "News about Reliance Industries",
        "IT sector developments",
        "Pharmaceutical companies performance",
    ]

    graph = get_query_graph()
    response_times = []

    for query in queries:
        start_time = time.time()
        state = create_query_state(query)
        final_state = graph.invoke(state)
        end_time = time.time()

        response_time = end_time - start_time
        response_times.append(response_time)

        results_count = len(final_state.get("results", []))
        logger.info(f"  Query: '{query[:50]}...' - {response_time:.2f}s - {results_count} results")

    avg_time = sum(response_times) / len(response_times) if response_times else 0
    max_time = max(response_times) if response_times else 0

    logger.info(f"Query Performance Results:")
    logger.info(f"  Average response time: {avg_time:.2f}s")
    logger.info(f"  Maximum response time: {max_time:.2f}s")
    logger.info(f"  Target: <2s")

    return avg_time < 2.0  # Target: <2s


async def main():
    """Run all benchmarks."""
    logger.info("=" * 50)
    logger.info("Starting Benchmark Suite")
    logger.info("=" * 50)

    results = {}

    # Deduplication
    results["deduplication"] = await benchmark_deduplication()
    logger.info("")

    # Entity extraction
    results["entity_extraction"] = await benchmark_entity_extraction()
    logger.info("")

    # Query performance
    results["query_performance"] = await benchmark_query_performance()
    logger.info("")

    # Summary
    logger.info("=" * 50)
    logger.info("Benchmark Summary")
    logger.info("=" * 50)
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {test}: {status}")

    all_passed = all(results.values())
    logger.info("=" * 50)
    logger.info(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    logger.info("=" * 50)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

