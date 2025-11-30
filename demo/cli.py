"""CLI interface for the financial news intelligence system."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.graph import get_news_processing_graph, get_query_graph
from src.core.state import create_article_state, create_query_state
from src.database.sql_db import get_sql_db

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def print_article(article_id: int, article_data: dict):
    """Print article information."""
    print("\n" + "=" * 60)
    print(f"Article ID: {article_id}")
    print(f"Title: {article_data.get('title', 'N/A')}")
    print(f"Source: {article_data.get('source', 'N/A')}")
    print(f"Timestamp: {article_data.get('timestamp', 'N/A')}")
    print(f"Content: {article_data.get('content', '')[:200]}...")
    print("=" * 60)


def print_query_results(results: list):
    """Print query results."""
    if not results:
        print("No results found.")
        return

    print(f"\nFound {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('title', 'N/A')}")
        print(f"   Similarity: {result.get('similarity_score', 0):.3f}")
        print(f"   Source: {result.get('source', 'N/A')}")
        print(f"   Stock Impacts: {len(result.get('stock_impacts', []))}")
        print()


def ingest_command(args):
    """Handle ingest command."""
    article = {
        "title": args.title,
        "content": args.content,
        "source": args.source or "CLI",
    }

    graph = get_news_processing_graph()
    state = create_article_state(article)
    final_state = graph.invoke(state)

    if final_state.get("errors"):
        print(f"Errors: {', '.join(final_state['errors'])}")
        return

    if final_state.get("is_duplicate"):
        print(f"Article is a duplicate of article ID: {final_state.get('duplicate_of_id')}")
    else:
        article_id = final_state.get("article_id")
        print(f"Article ingested successfully! ID: {article_id}")
        print(f"Entities extracted: {len(final_state.get('entities', []))}")
        print(f"Stock impacts: {len(final_state.get('stock_impacts', []))}")


def query_command(args):
    """Handle query command."""
    graph = get_query_graph()
    state = create_query_state(args.query)
    final_state = graph.invoke(state)

    if final_state.get("errors"):
        print(f"Errors: {', '.join(final_state['errors'])}")
        return

    results = final_state.get("results", [])
    print_query_results(results)


async def list_command(args):
    """Handle list command."""
    sql_db = get_sql_db()
    articles = await sql_db.get_recent_articles(limit=args.limit)

    if not articles:
        print("No articles found.")
        return

    print(f"\nRecent articles (showing {len(articles)}):\n")
    for article in articles:
        print(f"  [{article.id}] {article.title[:60]}...")
        print(f"      Source: {article.source}, Date: {article.timestamp}")
        print()


async def stats_command(args):
    """Handle stats command."""
    sql_db = get_sql_db()
    articles = await sql_db.get_recent_articles(limit=1000)

    total = len(articles)
    duplicates = sum(1 for a in articles if a.is_duplicate)
    unique = total - duplicates

    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"Total articles: {total}")
    print(f"Unique articles: {unique}")
    print(f"Duplicates: {duplicates}")
    print("=" * 60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Financial News Intelligence System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a news article")
    ingest_parser.add_argument("--title", required=True, help="Article title")
    ingest_parser.add_argument("--content", required=True, help="Article content")
    ingest_parser.add_argument("--source", help="Article source")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query news articles")
    query_parser.add_argument("query", help="Query string")

    # List command
    list_parser = subparsers.add_parser("list", help="List recent articles")
    list_parser.add_argument("--limit", type=int, default=10, help="Number of articles to show")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "ingest":
            ingest_command(args)
        elif args.command == "query":
            query_command(args)
        elif args.command == "list":
            asyncio.run(list_command(args))
        elif args.command == "stats":
            asyncio.run(stats_command(args))
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

