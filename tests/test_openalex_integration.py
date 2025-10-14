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

if __name__ == "__main__":
    print("\n" + "="*60)
    print("OPENALEX INTEGRATION TEST SUITE")
    print("="*60)

    tests = [
        ("Block Extraction", test_extraction),
        ("API Execution", test_execution),
        ("Result Formatting", test_formatting),
        ("Log Generation", test_logging),
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
