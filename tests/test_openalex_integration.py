#!/usr/bin/env python3
"""
Test script for OpenAlex integration.

Tests the full flow of:
1. Extracting <OPENALEX> blocks from text
2. Executing API calls
3. Formatting results
"""

from pathlib import Path
from llm_maths_research.utils.openalex_blocks import (
    extract_openalex_blocks,
    execute_openalex_calls,
    format_openalex_results,
    log_openalex_calls,
)

# Test text with sample <OPENALEX> block
test_response = """
Here is my analysis of the problem.

<OPENALEX>
[
  {
    "function": "search_literature",
    "arguments": {
      "query": "Turing patterns",
      "max_results": 3
    },
    "purpose": "Find seminal work on Turing patterns"
  },
  {
    "function": "get_paper",
    "arguments": {
      "identifier": "W2100837269"
    },
    "purpose": "Read Turing's original 1952 paper"
  }
]
</OPENALEX>

Based on these results, I propose the following approach...
"""

def test_extraction():
    """Test that <OPENALEX> blocks are correctly extracted."""
    print("="*60)
    print("TEST 1: Extract OpenAlex Blocks")
    print("="*60)

    calls = extract_openalex_blocks(test_response)

    if calls is None:
        print("✗ FAILED: No calls extracted")
        return False

    if len(calls) != 2:
        print(f"✗ FAILED: Expected 2 calls, got {len(calls)}")
        return False

    print(f"✓ Extracted {len(calls)} API calls")
    for i, call in enumerate(calls):
        print(f"  Call {i+1}: {call['function']} - {call['purpose']}")

    return True

def test_execution():
    """Test that API calls are executed and formatted."""
    print("\n" + "="*60)
    print("TEST 2: Execute API Calls")
    print("="*60)

    calls = extract_openalex_blocks(test_response)

    if calls is None:
        print("✗ FAILED: No calls to execute")
        return False

    # Execute calls
    results = execute_openalex_calls(
        calls,
        session_dir=Path("./test_cache"),
        email=None
    )

    if not results:
        print("✗ FAILED: No results returned")
        return False

    print(f"✓ Executed {len(results)} API calls")

    # Check results
    for result in results:
        status = "✓" if result['success'] else "✗"
        function = result['function']
        print(f"  {status} {function}")

        if result['success']:
            if function == 'search_literature':
                num_papers = len(result['result'].get('results', []))
                print(f"      Found {num_papers} papers")
            elif function == 'get_paper':
                title = result['result'].get('title', 'Unknown')[:50]
                print(f"      Retrieved: {title}")
        else:
            print(f"      Error: {result.get('error', 'Unknown')}")

    return all(r['success'] for r in results)

def test_formatting():
    """Test that results are formatted for prompt inclusion."""
    print("\n" + "="*60)
    print("TEST 3: Format Results for Prompt")
    print("="*60)

    calls = extract_openalex_blocks(test_response)
    results = execute_openalex_calls(
        calls,
        session_dir=Path("./test_cache"),
        email=None
    )

    formatted = format_openalex_results(results)

    if not formatted:
        print("✗ FAILED: No formatted output")
        return False

    print("✓ Formatted results:")
    print("-" * 60)
    print(formatted[:500])
    if len(formatted) > 500:
        print(f"... ({len(formatted) - 500} more characters)")
    print("-" * 60)

    return True

def test_logging():
    """Test that concise logs are generated."""
    print("\n" + "="*60)
    print("TEST 4: Generate Log Summary")
    print("="*60)

    calls = extract_openalex_blocks(test_response)
    results = execute_openalex_calls(
        calls,
        session_dir=Path("./test_cache"),
        email=None
    )

    log_summary = log_openalex_calls(results)

    print("✓ Log summary:")
    print(log_summary)

    return True


def test_arxiv_integration():
    """Test ArXiv paper download through the full <OPENALEX> workflow."""
    print("\n" + "="*60)
    print("TEST 5: ArXiv Download Integration")
    print("="*60)

    # Simulate agent response with ArXiv download request
    arxiv_response = """
<OPENALEX>
[
  {
    "function": "get_arxiv_paper",
    "arguments": {
      "arxiv_id": "1706.03762"
    },
    "purpose": "Download Transformer paper for detailed analysis"
  }
]
</OPENALEX>
"""

    # Step 1: Extract blocks
    calls = extract_openalex_blocks(arxiv_response)

    if not calls or len(calls) != 1:
        print(f"✗ FAILED: Expected 1 call, got {len(calls) if calls else 0}")
        return False

    if calls[0]['function'] != 'get_arxiv_paper':
        print(f"✗ FAILED: Expected get_arxiv_paper, got {calls[0]['function']}")
        return False

    print("✓ Extracted ArXiv download request")

    # Step 2: Execute the ArXiv download
    results = execute_openalex_calls(
        calls,
        session_dir=Path("./test_cache"),
        email=None
    )

    if not results or len(results) != 1:
        print(f"✗ FAILED: Expected 1 result, got {len(results) if results else 0}")
        return False

    result = results[0]

    if not result.get('success'):
        error = result.get('error', 'Unknown error')
        print(f"✗ FAILED: ArXiv download failed: {error}")
        return False

    # Verify result structure
    arxiv_result = result['result']
    if arxiv_result.get('arxiv_id') != "1706.03762":
        print(f"✗ FAILED: ArXiv ID mismatch")
        return False

    if arxiv_result.get('char_count', 0) < 1000:
        print(f"✗ FAILED: Content too short ({arxiv_result.get('char_count', 0)} chars)")
        return False

    print(f"✓ Downloaded ArXiv paper {arxiv_result['arxiv_id']}")
    print(f"  Character count: {arxiv_result['char_count']:,}")
    print(f"  Approximate tokens: {arxiv_result['approx_tokens']:,}")

    # Step 3: Format results for prompt
    formatted = format_openalex_results(results)

    if not formatted:
        print("✗ FAILED: No formatted output")
        return False

    # Verify formatting includes key information
    required_strings = [
        "ArXiv Paper: 1706.03762",
        "Character count:",
        "Approximate tokens:",
    ]

    for required in required_strings:
        if required not in formatted:
            print(f"✗ FAILED: Formatted output missing '{required}'")
            return False

    # Verify paper content is included
    if 'attention' not in formatted.lower():
        print("✗ FAILED: Paper content not included in formatted output")
        return False

    print("✓ Formatted ArXiv results for prompt inclusion:")
    print("-" * 60)
    print(formatted[:400])
    if len(formatted) > 400:
        print(f"... ({len(formatted) - 400} more characters)")
    print("-" * 60)

    # Step 4: Verify log generation
    log_summary = log_openalex_calls(results)

    if 'get_arxiv_paper' not in log_summary:
        print("✗ FAILED: Log summary missing get_arxiv_paper")
        return False

    if '1706.03762' not in log_summary:
        print("✗ FAILED: Log summary missing ArXiv ID")
        return False

    print("✓ Log summary generated:")
    print(log_summary)

    print("\n✓ ArXiv integration test passed!")
    print("  The agent can successfully download ArXiv papers through <OPENALEX> blocks")

    return True

if __name__ == "__main__":
    print("\n" + "="*60)
    print("OPENALEX INTEGRATION TEST SUITE")
    print("="*60)

    tests = [
        ("Block Extraction", test_extraction),
        ("API Execution", test_execution),
        ("Result Formatting", test_formatting),
        ("Log Generation", test_logging),
        ("ArXiv Integration", test_arxiv_integration),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Integration is working correctly.")
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review errors above.")
