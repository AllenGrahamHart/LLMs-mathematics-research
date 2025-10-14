"""
Test ArXiv paper download functionality.
"""

import tempfile
from pathlib import Path
from llm_maths_research.literature import get_arxiv_paper


def test_arxiv_paper_download():
    """Test downloading and extracting a well-known ArXiv paper."""
    # Test with "Attention Is All You Need" (Transformer paper)
    arxiv_id = "1706.03762"

    # Use temporary directory for cache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        # Download paper
        result = get_arxiv_paper(arxiv_id, session_dir=cache_dir)

        # Check success
        assert result is not None, "Result should not be None"
        assert result.get('success') is True, f"Download failed: {result.get('error')}"
        assert result.get('arxiv_id') == arxiv_id, "ArXiv ID mismatch"

        # Check content is extracted
        assert 'content' in result, "Content should be present"
        assert len(result['content']) > 0, "Content should not be empty"
        assert result['char_count'] > 1000, "Content should have substantial length"
        assert result['approx_tokens'] > 250, "Should have reasonable token count"

        # Check expected content (abstract mentions "attention mechanism")
        content_lower = result['content'].lower()
        assert 'attention' in content_lower, "Paper content should mention 'attention'"

        print(f"✓ Successfully downloaded {arxiv_id}")
        print(f"  Character count: {result['char_count']:,}")
        print(f"  Approximate tokens: {result['approx_tokens']:,}")
        print(f"  Content preview: {result['content'][:100]}...")


def test_arxiv_invalid_id():
    """Test handling of invalid ArXiv ID."""
    arxiv_id = "9999.99999"  # Non-existent paper

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        result = get_arxiv_paper(arxiv_id, session_dir=cache_dir)

        # Should fail gracefully
        assert result is not None, "Result should not be None"
        assert result.get('success') is False, "Should fail for invalid ID"
        assert 'error' in result, "Should contain error message"

        print(f"✓ Correctly handled invalid ArXiv ID: {result['error'][:60]}...")


if __name__ == "__main__":
    # Allow running directly for quick testing
    print("Running ArXiv download tests...\n")
    test_arxiv_paper_download()
    print()
    test_arxiv_invalid_id()
    print("\n✓ All ArXiv download tests passed!")
