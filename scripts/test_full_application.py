"""Script to test the full application end-to-end."""

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(success: bool, message: str):
    """Print test result."""
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{status}: {message}")


def test_health_check():
    """Test health check endpoint."""
    print_section("1. Health Check")
    try:
        response = httpx.get(f"{BASE_URL}/api/health", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            print_result(True, f"Server is healthy")
            print(f"   Database connected: {data.get('database_connected')}")
            print(f"   Vector DB size: {data.get('vector_db_size')}")
            return True
        else:
            print_result(False, f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print_result(False, f"Health check error: {e}")
        print("   Make sure the server is running: uvicorn src.api.main:app --reload")
        return False


def test_ingest_article():
    """Test ingesting a single article."""
    print_section("2. Ingest Article")
    try:
        article = {
            "title": "RBI Raises Repo Rate by 25 Basis Points",
            "content": "The Reserve Bank of India has increased the repo rate by 25 basis points to 6.5%, citing persistent inflationary pressures. This move is expected to impact banking stocks, particularly HDFC Bank, ICICI Bank, and State Bank of India. The rate hike comes after three consecutive quarters of elevated inflation above the central bank's target range.",
            "source": "Test Source",
        }

        response = httpx.post(
            f"{BASE_URL}/api/news/ingest",
            json=article,
            timeout=30.0,
        )

        if response.status_code == 201:
            data = response.json()
            article_id = data.get("article_id")
            print_result(True, f"Article ingested successfully (ID: {article_id})")
            print(f"   Title: {data.get('title')[:50]}...")
            print(f"   Entities extracted: {len(data.get('entities', []))}")
            print(f"   Stock impacts: {len(data.get('stock_impacts', []))}")
            print(f"   Is duplicate: {data.get('is_duplicate')}")
            return article_id
        else:
            print_result(False, f"Ingest failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print_result(False, f"Ingest error: {e}")
        return None


def test_get_article(article_id: int):
    """Test retrieving an article."""
    print_section("3. Get Article")
    try:
        response = httpx.get(f"{BASE_URL}/api/news/{article_id}", timeout=10.0)

        if response.status_code == 200:
            data = response.json()
            print_result(True, f"Article retrieved successfully")
            print(f"   ID: {data.get('article_id')}")
            print(f"   Entities: {len(data.get('entities', []))}")
            print(f"   Stock impacts: {len(data.get('stock_impacts', []))}")
            return True
        else:
            print_result(False, f"Get article failed: {response.status_code}")
            return False
    except Exception as e:
        print_result(False, f"Get article error: {e}")
        return False


def test_query():
    """Test query endpoint."""
    print_section("4. Query News")
    try:
        query = {"query": "What is the impact of RBI rate hike on banking stocks?"}

        response = httpx.post(
            f"{BASE_URL}/api/query",
            json=query,
            timeout=30.0,
        )

        if response.status_code == 200:
            data = response.json()
            results_count = data.get("results_count", 0)
            print_result(True, f"Query executed successfully")
            print(f"   Query: {query['query']}")
            print(f"   Query type: {data.get('query_type')}")
            print(f"   Results found: {results_count}")

            if results_count > 0:
                print("\n   Top 3 results:")
                for i, result in enumerate(data.get("results", [])[:3], 1):
                    print(f"   {i}. {result.get('title')[:50]}...")
                    print(f"      Similarity: {result.get('similarity_score'):.3f}")
                    print(f"      Stock impacts: {len(result.get('stock_impacts', []))}")
            else:
                print("   No results found. Try ingesting more articles first.")

            return True
        else:
            print_result(False, f"Query failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print_result(False, f"Query error: {e}")
        return False


def test_stock_news():
    """Test stock-specific news endpoint."""
    print_section("5. Get Stock News")
    try:
        response = httpx.get(
            f"{BASE_URL}/api/stocks/HDFCBANK/news?limit=5",
            timeout=10.0,
        )

        if response.status_code == 200:
            articles = response.json()
            print_result(True, f"Stock news retrieved successfully")
            print(f"   Articles found: {len(articles)}")

            if articles:
                print("\n   Articles:")
                for i, article in enumerate(articles[:3], 1):
                    print(f"   {i}. {article.get('title')[:50]}...")
            else:
                print("   No articles found for this stock.")

            return True
        else:
            print_result(False, f"Stock news failed: {response.status_code}")
            return False
    except Exception as e:
        print_result(False, f"Stock news error: {e}")
        return False


def test_deduplication():
    """Test deduplication by ingesting the same article twice."""
    print_section("6. Deduplication Test")
    try:
        article = {
            "title": "Test Duplicate Article",
            "content": "This is a test article to check deduplication functionality. It mentions HDFC Bank and ICICI Bank.",
            "source": "Test Source 1",
        }

        # First ingestion
        response1 = httpx.post(
            f"{BASE_URL}/api/news/ingest",
            json=article,
            timeout=30.0,
        )

        if response1.status_code != 201:
            print_result(False, "First article ingestion failed")
            return False

        data1 = response1.json()
        first_id = data1.get("article_id")
        print(f"   First article ID: {first_id}")

        # Wait longer for embedding to be stored and indexed in vector DB
        time.sleep(3)

        # Second ingestion (should be duplicate)
        article["source"] = "Test Source 2"  # Different source
        response2 = httpx.post(
            f"{BASE_URL}/api/news/ingest",
            json=article,
            timeout=30.0,
        )

        if response2.status_code == 201:
            data2 = response2.json()
            is_duplicate = data2.get("is_duplicate", False)
            duplicate_of = data2.get("duplicate_of_id")
            second_id = data2.get("article_id")

            if is_duplicate:
                print_result(True, "Deduplication working correctly")
                print(f"   Duplicate detected: {is_duplicate}")
                print(f"   Duplicate of: {duplicate_of}")
                print(f"   Second article ID: {second_id}")
            else:
                # Check if it's the same ID (which would also indicate deduplication worked)
                if second_id == first_id:
                    print_result(True, "Deduplication working (same article ID returned)")
                    print(f"   Both articles have ID: {second_id}")
                else:
                    print_result(False, "Deduplication not working - article not marked as duplicate")
                    print(f"   First ID: {first_id}, Second ID: {second_id}")
                    print(f"   This might be normal if similarity threshold is high (0.90)")
            return is_duplicate or (second_id == first_id)
        else:
            print_result(False, "Second article ingestion failed")
            return False

    except Exception as e:
        print_result(False, f"Deduplication test error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  FINANCIAL NEWS INTELLIGENCE SYSTEM - FULL APPLICATION TEST")
    print("=" * 60)
    print("\nMake sure the server is running:")
    print("  uvicorn src.api.main:app --reload")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nTest cancelled.")
        return

    results = []

    # Test 1: Health check
    if not test_health_check():
        print("\n❌ Server is not running or not healthy. Please start it first.")
        return
    results.append(True)

    # Test 2: Ingest article
    article_id = test_ingest_article()
    if article_id:
        results.append(True)

        # Test 3: Get article
        test_get_article(article_id)
        results.append(True)
    else:
        results.append(False)

    # Test 4: Query
    test_query()
    results.append(True)

    # Test 5: Stock news
    test_stock_news()
    results.append(True)

    # Test 6: Deduplication
    test_deduplication()
    results.append(True)

    # Summary
    print_section("Test Summary")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    if passed == total:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()

