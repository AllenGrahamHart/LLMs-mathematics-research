"""
OpenAlex block extraction and execution utilities.

Handles parsing of <OPENALEX> JSON blocks from researcher/critic responses,
executes API calls, and formats results for prompt inclusion.
"""

import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path


def extract_openalex_blocks(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Extract and parse <OPENALEX> JSON blocks from response text.

    Args:
        text: Response text that may contain <OPENALEX>...</OPENALEX> blocks

    Returns:
        List of API call dictionaries, or None if no valid blocks found

    Example block format:
        <OPENALEX>
        [
          {
            "function": "search_literature",
            "arguments": {"query": "pattern formation", "max_results": 10},
            "purpose": "Find recent work"
          }
        ]
        </OPENALEX>
    """
    # Find all <OPENALEX>...</OPENALEX> blocks
    pattern = r'<OPENALEX>\s*(.*?)\s*</OPENALEX>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    if not matches:
        return None

    all_calls = []

    for match in matches:
        try:
            # Parse JSON
            calls = json.loads(match.strip())

            # Ensure it's a list
            if not isinstance(calls, list):
                print(f"Warning: <OPENALEX> block contains non-list JSON: {type(calls)}")
                continue

            # Validate each call has required fields
            for call in calls:
                if not isinstance(call, dict):
                    print(f"Warning: Skipping non-dict API call: {call}")
                    continue

                if 'function' not in call or 'arguments' not in call:
                    print(f"Warning: Skipping incomplete API call (missing function or arguments): {call}")
                    continue

                # Add to results
                all_calls.append(call)

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse <OPENALEX> JSON block: {e}")
            print(f"Block content: {match[:200]}...")
            continue

    return all_calls if all_calls else None


def execute_openalex_calls(
    calls: List[Dict[str, Any]],
    session_dir: Path,
    email: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Execute a list of OpenAlex API calls and return formatted results.

    Args:
        calls: List of API call dictionaries with 'function', 'arguments', 'purpose'
        session_dir: Session directory for caching
        email: Email for polite API usage

    Returns:
        List of result dictionaries with 'call', 'success', 'result'/'error'
    """
    from ..literature import search_literature, get_paper

    results = []

    for i, call in enumerate(calls):
        function_name = call.get('function')
        arguments = call.get('arguments', {})
        purpose = call.get('purpose', 'No purpose specified')

        result_entry = {
            'call_index': i + 1,
            'function': function_name,
            'purpose': purpose,
            'success': False
        }

        try:
            # Execute the appropriate function
            if function_name == 'search_literature':
                # Add session_dir and email to arguments
                arguments['session_dir'] = session_dir
                if email:
                    arguments['email'] = email

                api_result = search_literature(**arguments)

                # Check for API error
                if api_result.get('error'):
                    result_entry['error'] = api_result.get('message', 'Unknown error')
                else:
                    result_entry['success'] = True
                    result_entry['result'] = api_result

            elif function_name == 'get_paper':
                # Add session_dir and email to arguments
                arguments['session_dir'] = session_dir
                if email:
                    arguments['email'] = email

                api_result = get_paper(**arguments)

                # Check for API error
                if api_result.get('error'):
                    result_entry['error'] = api_result.get('message', 'Unknown error')
                else:
                    result_entry['success'] = True
                    result_entry['result'] = api_result

            else:
                result_entry['error'] = f"Unknown function: {function_name}"

        except Exception as e:
            result_entry['error'] = f"Exception during execution: {str(e)}"

        results.append(result_entry)

    return results


def format_openalex_results(results: List[Dict[str, Any]], max_results_per_call: int = 10) -> str:
    """
    Format OpenAlex API results for inclusion in prompts.

    Args:
        results: List of result dictionaries from execute_openalex_calls()
        max_results_per_call: Maximum number of papers to include per search call

    Returns:
        Formatted string suitable for prompt inclusion
    """
    if not results:
        return "No OpenAlex API calls were made."

    sections = []

    for result_entry in results:
        call_index = result_entry['call_index']
        function = result_entry['function']
        purpose = result_entry['purpose']

        section = f"=== Call #{call_index}: {function} ===\n"
        section += f"Purpose: {purpose}\n\n"

        if not result_entry['success']:
            section += f"ERROR: {result_entry['error']}\n"
            sections.append(section)
            continue

        api_result = result_entry['result']

        # Format based on function type
        if function == 'search_literature':
            query_info = api_result.get('query_info', {})
            papers = api_result.get('results', [])

            section += f"Query: {query_info.get('search_query', 'N/A')}\n"
            section += f"Filters: {query_info.get('filters_applied', {})}\n"
            section += f"Total results: {query_info.get('total_results', 0)}\n"
            section += f"Returned: {len(papers)} papers\n\n"

            # Format each paper (limit to max_results_per_call)
            for i, paper in enumerate(papers[:max_results_per_call]):
                section += f"  [{i+1}] {paper['title']}\n"
                section += f"      Authors: {paper['author_string']}\n"
                section += f"      Year: {paper['publication_year']} | Venue: {paper['venue']}\n"
                section += f"      Citations: {paper['cited_by_count']} | ID: {paper['id']}\n"
                if paper.get('doi'):
                    section += f"      DOI: {paper['doi']}\n"
                if paper.get('abstract_snippet'):
                    section += f"      Abstract: {paper['abstract_snippet']}\n"
                section += "\n"

            if len(papers) > max_results_per_call:
                section += f"  ... and {len(papers) - max_results_per_call} more results (truncated)\n"

        elif function == 'get_paper':
            section += f"Title: {api_result.get('title', 'N/A')}\n"
            section += f"Authors: {api_result.get('author_string', 'N/A')}\n"
            section += f"Year: {api_result.get('publication_year', 'N/A')}\n"
            section += f"Venue: {api_result.get('venue', {}).get('name', 'N/A')}\n"
            section += f"Citations: {api_result.get('cited_by_count', 0)}\n"

            if api_result.get('doi'):
                section += f"DOI: {api_result['doi']}\n"

            if api_result.get('abstract'):
                # Include full abstract (no truncation)
                abstract = api_result['abstract']
                section += f"\nAbstract:\n{abstract}\n"

            # Include reference count
            section += f"\nReferences: {api_result.get('references_count', 0)} papers\n"

            # Include first few references
            references = api_result.get('references', [])
            if references:
                section += f"First {min(3, len(references))} references:\n"
                for i, ref in enumerate(references[:3]):
                    section += f"  [{i+1}] {ref.get('title', 'Unknown')} ({ref.get('year', 'N/A')})\n"
                if len(references) > 3:
                    section += f"  ... and {len(references) - 3} more\n"

            # Include formatted citations
            citations = api_result.get('formatted_citations', {})
            if citations.get('apa'):
                section += f"\nAPA Citation:\n{citations['apa']}\n"

        sections.append(section)

    return "\n".join(sections)


def log_openalex_calls(results: List[Dict[str, Any]]) -> str:
    """
    Create a concise log entry for OpenAlex API calls.

    Args:
        results: List of result dictionaries from execute_openalex_calls()

    Returns:
        Formatted log string
    """
    if not results:
        return "No OpenAlex calls"

    log_lines = []
    for result_entry in results:
        function = result_entry['function']
        success = "✓" if result_entry['success'] else "✗"

        if result_entry['success']:
            if function == 'search_literature':
                num_results = len(result_entry['result'].get('results', []))
                log_lines.append(f"  {success} search_literature: {num_results} results")
            elif function == 'get_paper':
                title = result_entry['result'].get('title', 'Unknown')[:50]
                log_lines.append(f"  {success} get_paper: {title}")
        else:
            error = result_entry.get('error', 'Unknown error')[:50]
            log_lines.append(f"  {success} {function}: {error}")

    return "\n".join(log_lines)
