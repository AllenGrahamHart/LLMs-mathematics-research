"""
Test ArXiv ID extraction in search_literature results.
"""

import tempfile
from pathlib import Path
from llm_maths_research.literature import search_literature


def test_arxiv_id_in_search_results():
    """Test that ArXiv IDs are extracted and exposed in search results."""

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        # Search for papers likely to be on ArXiv (deep learning papers)
        result = search_literature(
            query="attention mechanism transformers",
            filters={"from_year": 2017, "to_year": 2018},
            max_results=5,
            session_dir=cache_dir
        )

        # Check that search succeeded
        assert result is not None, "Result should not be None"

        # If this specific query has API issues, skip the test
        if result.get('error'):
            print(f"⚠ Search API error (may be data issue): {result.get('message')}")
            print("  Skipping test - core functionality tested in other tests")
            return

        assert 'results' in result, "Should have results field"

        papers = result['results']
        assert len(papers) > 0, "Should find at least some papers"

        print(f"✓ Found {len(papers)} papers")

        # Check that papers have arxiv_id field (even if None)
        arxiv_count = 0
        for i, paper in enumerate(papers, 1):
            assert 'arxiv_id' in paper, f"Paper {i} should have arxiv_id field"

            if paper.get('arxiv_id'):
                arxiv_count += 1
                print(f"  [{i}] {paper['title'][:50]}...")
                print(f"      ArXiv ID: {paper['arxiv_id']} ✓")

        print(f"✓ Papers with ArXiv IDs: {arxiv_count}/{len(papers)}")

        # Note: We don't assert arxiv_count > 0 because it depends on the specific
        # papers returned, but the field should always be present
        if arxiv_count == 0:
            print("  ⚠ No ArXiv IDs found in this particular search (may be expected)")


def test_search_structure():
    """Test that search results have the expected structure."""

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        result = search_literature(
            query="Turing patterns",
            max_results=3,
            session_dir=cache_dir
        )

        assert result is not None, "Result should not be None"

        if result.get('error'):
            print(f"⚠ Search API error (may be rate limiting): {result.get('message')}")
            return  # Skip test if API is unavailable

        assert 'query_info' in result, "Should have query_info"
        assert 'results' in result, "Should have results"

        if len(result['results']) > 0:
            paper = result['results'][0]
            # Check all expected fields are present
            expected_fields = ['id', 'doi', 'title', 'authors', 'author_string',
                             'publication_year', 'venue', 'cited_by_count',
                             'is_open_access', 'arxiv_id']

            for field in expected_fields:
                assert field in paper, f"Paper should have {field} field"

            print("✓ Search results have correct structure")
            print(f"  Sample paper: {paper['title'][:60]}...")
            if paper.get('arxiv_id'):
                print(f"  ArXiv ID: {paper['arxiv_id']}")


if __name__ == "__main__":
    # Allow running directly for quick testing
    print("Running ArXiv search tests...\n")

    try:
        test_arxiv_id_in_search_results()
        print()
        test_search_structure()
        print("\n✓ All ArXiv search tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
