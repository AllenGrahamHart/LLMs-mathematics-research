"""
Tests for literature search functionality.

Note: These tests make real API calls to OpenAlex. They should be used sparingly
to avoid hitting rate limits. Consider mocking for CI/CD.
"""

import pytest
from pathlib import Path
from llm_maths_research.literature import search_literature, get_paper, cleanup


class TestSearchLiterature:
    """Tests for search_literature function."""

    def test_basic_keyword_search(self):
        """Test basic keyword search returns results."""
        results = search_literature(
            query="Turing patterns",
            max_results=5
        )

        assert 'results' in results
        assert 'query_info' in results
        assert len(results['results']) <= 5
        assert results['query_info']['search_query'] == "Turing patterns"

    def test_search_with_filters(self):
        """Test search with date and citation filters."""
        results = search_literature(
            query="bifurcation theory",
            filters={"from_year": 2020, "min_citations": 10},
            max_results=5
        )

        assert 'results' in results
        # Check that results respect filters
        for paper in results['results']:
            if paper['publication_year']:
                assert paper['publication_year'] >= 2020
            if paper['cited_by_count']:
                assert paper['cited_by_count'] >= 10

    def test_citation_network_search(self):
        """Test finding papers that cite a specific work."""
        # Turing's 1952 paper: W2100837269
        results = search_literature(
            filters={"cites": "W2100837269"},
            max_results=10
        )

        assert 'results' in results
        assert len(results['results']) > 0
        # Papers citing Turing should be more recent
        for paper in results['results']:
            if paper['publication_year']:
                assert paper['publication_year'] >= 1952

    def test_doi_lookup(self):
        """Test looking up a specific paper by DOI."""
        # Example DOI - replace with a known valid one
        results = search_literature(
            filters={"doi": "10.1098/rstb.1952.0012"}  # Turing's paper
        )

        assert 'results' in results
        if len(results['results']) > 0:
            assert results['results'][0]['doi'] is not None

    def test_empty_search(self):
        """Test that empty search returns gracefully."""
        results = search_literature(max_results=5)

        # Should return some results even without query
        assert 'results' in results
        assert 'query_info' in results

    def test_no_results(self):
        """Test search with impossible constraints."""
        results = search_literature(
            query="zxcvbnmasdfghjkl",  # Gibberish
            filters={"from_year": 2050},  # Future date
            max_results=5
        )

        assert 'results' in results
        assert len(results['results']) == 0


class TestGetPaper:
    """Tests for get_paper function."""

    def test_get_paper_by_openalex_id(self):
        """Test retrieving a paper by OpenAlex ID."""
        # Turing's 1952 paper
        paper = get_paper("W2100837269")

        assert 'title' in paper
        assert 'authors' in paper
        assert 'abstract' in paper or 'abstract' in paper  # Abstract might not always be available
        assert paper['id'] == "W2100837269"

    def test_get_paper_by_doi(self):
        """Test retrieving a paper by DOI."""
        paper = get_paper("10.1098/rstb.1952.0012")  # Turing's paper

        assert 'title' in paper
        assert 'doi' in paper
        assert '10.1098/rstb.1952.0012' in paper['doi']

    def test_paper_has_required_fields(self):
        """Test that retrieved paper has all expected fields."""
        paper = get_paper("W2100837269")

        required_fields = [
            'id', 'title', 'authors', 'publication_year',
            'venue', 'cited_by_count', 'references', 'formatted_citations'
        ]

        for field in required_fields:
            assert field in paper, f"Missing required field: {field}"

    def test_formatted_citations(self):
        """Test that formatted citations are generated."""
        paper = get_paper("W2100837269")

        assert 'formatted_citations' in paper
        assert 'apa' in paper['formatted_citations']
        assert 'bibtex' in paper['formatted_citations']
        assert len(paper['formatted_citations']['apa']) > 0
        assert len(paper['formatted_citations']['bibtex']) > 0

    def test_invalid_id(self):
        """Test handling of invalid paper ID."""
        paper = get_paper("W999999999999")

        assert 'error' in paper
        assert paper['error'] is True


class TestCaching:
    """Tests for caching functionality."""

    def test_cache_reuse(self, tmp_path):
        """Test that repeated searches use cache."""
        # First search
        results1 = search_literature(
            query="test query cache",
            max_results=3,
            session_dir=tmp_path
        )

        # Second search (should use cache)
        results2 = search_literature(
            query="test query cache",
            max_results=3,
            session_dir=tmp_path
        )

        # Results should be identical
        assert results1 == results2

        # Check cache file exists
        cache_file = tmp_path / "literature_cache" / "openalex_cache.json"
        assert cache_file.exists()


def test_cleanup():
    """Test cleanup function runs without error."""
    cleanup()  # Should not raise


if __name__ == "__main__":
    # Run a quick test
    print("Testing basic search...")
    results = search_literature("Turing patterns", max_results=3)
    print(f"Found {len(results['results'])} results")

    if results['results']:
        print(f"\nFirst result: {results['results'][0]['title']}")

        print("\nTesting get_paper...")
        paper = get_paper(results['results'][0]['id'])
        print(f"Retrieved: {paper['title']}")
        print(f"Authors: {paper['author_string']}")
        print(f"Year: {paper['publication_year']}")
        print(f"Citations: {paper['cited_by_count']}")

    cleanup()
    print("\nBasic tests passed!")
