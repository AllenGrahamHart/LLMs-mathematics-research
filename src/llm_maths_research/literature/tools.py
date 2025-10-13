"""
Literature search tools for LLM researcher.

Provides simple, high-level functions for searching and retrieving scholarly literature
using the OpenAlex API.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from .openalex_client import OpenAlexClient


# Global client instance (initialized when first tool is called)
_client: Optional[OpenAlexClient] = None


def _get_client(session_dir: Optional[Path] = None, email: Optional[str] = None) -> OpenAlexClient:
    """
    Get or create the global OpenAlex client.

    Args:
        session_dir: Session directory for caching
        email: Email for polite API usage

    Returns:
        OpenAlexClient instance
    """
    global _client

    if _client is None:
        cache_dir = session_dir / "literature_cache" if session_dir else None
        _client = OpenAlexClient(email=email, cache_dir=cache_dir)

    return _client


def search_literature(
    query: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    max_results: int = 10,
    session_dir: Optional[Path] = None,
    email: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for scholarly literature using OpenAlex.

    This flexible function supports keyword search, citation network navigation,
    date filtering, and citation verification.

    Args:
        query: Search keywords (e.g., "pattern formation Turing")
               Supports boolean operators: AND, OR, NOT (must be uppercase)
        filters: Dictionary of filters:
            - cites (str): OpenAlex ID of work to find papers citing it
            - cited_by (str): OpenAlex ID of work to find papers it cites
            - from_year (int): Minimum publication year
            - to_year (int): Maximum publication year
            - publication_year (int): Exact publication year
            - min_citations (int): Minimum citation count
            - is_open_access (bool): Filter for open access papers
            - publication_type (str): "article", "preprint", "book-chapter", etc.
            - has_fulltext (bool): Filter for papers with fulltext available
            - doi (str): Exact DOI lookup
            - title (str): Search by title (fuzzy match)
            - authors (str): Search by author name
            - sort_by (str): "cited_by_count", "publication_date", "relevance_score"
        max_results: Maximum number of results to return (1-50)
        session_dir: Session directory for caching (optional)
        email: Email for API usage (optional but recommended)

    Returns:
        Dictionary with:
            - query_info: Information about the query
            - results: List of papers with metadata

    Examples:
        # Keyword search
        search_literature("Turing patterns bifurcation", max_results=20)

        # Search with filters
        search_literature(
            query="neural networks",
            filters={"from_year": 2022, "is_open_access": True}
        )

        # Citation network (papers citing a work)
        search_literature(filters={"cites": "W2741809807"})

        # Citation network (papers cited by a work)
        search_literature(filters={"cited_by": "W2741809807"})

        # Verify citation by DOI
        search_literature(filters={"doi": "10.1016/j.chaos.2023.113456"})

        # Verify citation by title + year
        search_literature(
            filters={"title": "Pattern Formation", "publication_year": 2023}
        )
    """
    client = _get_client(session_dir, email)

    try:
        # Make API request
        response = client.search_works(
            query=query,
            filters=filters,
            per_page=min(max_results, 50),
            page=1
        )

        # Format results
        results = []
        for work in response.get('results', []):
            # Extract author names
            authors = [
                auth.get('author', {}).get('display_name', 'Unknown')
                for auth in work.get('authorships', [])[:10]  # Limit to first 10
            ]

            # Get primary location (publication venue)
            primary_loc = work.get('primary_location', {})
            venue = primary_loc.get('source', {}).get('display_name', 'Unknown')

            # Get abstract snippet (from inverted index)
            abstract_inv = work.get('abstract_inverted_index', {})
            if abstract_inv:
                # Get first ~30 words for snippet
                word_positions = [(pos, word) for word, positions in abstract_inv.items() for pos in positions]
                word_positions.sort(key=lambda x: x[0])
                abstract_snippet = ' '.join(word for _, word in word_positions[:30])
                if len(word_positions) > 30:
                    abstract_snippet += "..."
            else:
                abstract_snippet = None

            # Get topics
            topics = [topic.get('display_name', '') for topic in work.get('topics', [])[:3]]

            # Format result
            result = {
                'id': work.get('id', '').split('/')[-1],  # Extract W123 from URL
                'doi': work.get('doi', '').replace('https://doi.org/', '') if work.get('doi') else None,
                'title': work.get('title', 'Untitled'),
                'authors': authors,
                'author_string': ', '.join(authors[:3]) + (' et al.' if len(authors) > 3 else ''),
                'publication_year': work.get('publication_year'),
                'publication_date': work.get('publication_date'),
                'venue': venue,
                'cited_by_count': work.get('cited_by_count', 0),
                'is_open_access': work.get('open_access', {}).get('is_oa', False),
                'abstract_snippet': abstract_snippet,
                'primary_topics': topics,
                'relevance_score': work.get('relevance_score')
            }
            results.append(result)

        # Build query info
        query_info = {
            'search_query': query,
            'filters_applied': filters or {},
            'total_results': response.get('meta', {}).get('count', 0),
            'returned_results': len(results)
        }

        return {
            'query_info': query_info,
            'results': results
        }

    except Exception as e:
        return {
            'error': True,
            'message': str(e),
            'query_info': {
                'search_query': query,
                'filters_applied': filters or {}
            },
            'results': []
        }


def get_paper(
    identifier: str,
    session_dir: Optional[Path] = None,
    email: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve full metadata and abstract for a specific paper.

    Args:
        identifier: One of:
            - OpenAlex ID: "W2741809807"
            - DOI: "10.1016/j.chaos.2023.113456"
            - OpenAlex URL: "https://openalex.org/W2741809807"
        session_dir: Session directory for caching (optional)
        email: Email for API usage (optional but recommended)

    Returns:
        Dictionary with complete paper metadata including:
            - id, doi, title
            - authors (with affiliations)
            - publication details
            - abstract (full text)
            - topics, keywords
            - citation metrics
            - references (papers this work cites)
            - formatted citations (APA, BibTeX)

    Examples:
        # By OpenAlex ID
        get_paper("W2741809807")

        # By DOI
        get_paper("10.1016/j.chaos.2023.113456")

        # By OpenAlex URL
        get_paper("https://openalex.org/W2741809807")
    """
    client = _get_client(session_dir, email)

    try:
        # Get work data
        work = client.get_work(identifier)

        # Extract detailed author information
        authors = []
        for authorship in work.get('authorships', []):
            author_data = authorship.get('author', {})
            institutions = [
                inst.get('display_name', 'Unknown')
                for inst in authorship.get('institutions', [])
            ]
            authors.append({
                'name': author_data.get('display_name', 'Unknown'),
                'position': authorship.get('author_position', 'unknown'),
                'institutions': institutions
            })

        # Get full abstract
        abstract = client.get_abstract(work)

        # Get venue information
        primary_loc = work.get('primary_location', {})
        venue_data = primary_loc.get('source', {})
        venue = {
            'name': venue_data.get('display_name', 'Unknown'),
            'type': venue_data.get('type', 'unknown')
        }

        # Get topics
        topics = [
            {
                'name': topic.get('display_name', ''),
                'score': topic.get('score', 0)
            }
            for topic in work.get('topics', [])
        ]

        # Get keywords
        keywords = work.get('keywords', [])
        if isinstance(keywords, list) and keywords:
            keywords = [kw.get('display_name', kw) if isinstance(kw, dict) else kw for kw in keywords]
        else:
            keywords = []

        # Get references
        references = []
        for ref_id in work.get('referenced_works', [])[:50]:  # Limit to first 50
            try:
                ref_work = client.get_work(ref_id)
                references.append({
                    'id': ref_id.split('/')[-1],
                    'title': ref_work.get('title', 'Unknown'),
                    'authors': ', '.join(auth.get('author', {}).get('display_name', '')
                                       for auth in ref_work.get('authorships', [])[:3]),
                    'year': ref_work.get('publication_year'),
                    'doi': ref_work.get('doi', '').replace('https://doi.org/', '') if ref_work.get('doi') else None
                })
            except:
                # If reference can't be fetched, just include the ID
                references.append({'id': ref_id.split('/')[-1]})

        # Get citation counts by year
        citations_by_year = work.get('counts_by_year', [])

        # Get open access information
        oa_info = work.get('open_access', {})
        oa_url = oa_info.get('oa_url')
        if not oa_url and primary_loc.get('pdf_url'):
            oa_url = primary_loc.get('pdf_url')

        # Generate formatted citations
        author_string = ', '.join(a['name'] for a in authors[:3])
        if len(authors) > 3:
            author_string += ' et al.'

        title = work.get('title', 'Untitled')
        year = work.get('publication_year', 'n.d.')
        doi = work.get('doi', '').replace('https://doi.org/', '') if work.get('doi') else None

        # APA style
        apa_citation = f"{author_string} ({year}). {title}. {venue['name']}."
        if doi:
            apa_citation += f" https://doi.org/{doi}"

        # BibTeX
        bibtex_key = f"{authors[0]['name'].split()[-1].lower()}{year}{title.split()[0].lower()}" if authors else f"unknown{year}"
        bibtex = f"""@article{{{bibtex_key},
  title={{{title}}},
  author={{{author_string}}},
  journal={{{venue['name']}}},
  year={{{year}}}"""
        if doi:
            bibtex += f",\n  doi={{{doi}}}"
        bibtex += "\n}"

        # Build result
        result = {
            'id': work.get('id', '').split('/')[-1],
            'doi': doi,
            'openalex_url': work.get('id', ''),
            'title': title,
            'authors': authors,
            'author_string': author_string,
            'publication_year': year,
            'publication_date': work.get('publication_date'),
            'venue': venue,
            'abstract': abstract,
            'topics': topics,
            'keywords': keywords,
            'cited_by_count': work.get('cited_by_count', 0),
            'citation_percentile': work.get('cited_by_percentile_year', {}).get('max'),
            'citations_by_year': citations_by_year,
            'references': references,
            'references_count': len(work.get('referenced_works', [])),
            'is_open_access': oa_info.get('is_oa', False),
            'oa_url': oa_url,
            'license': work.get('license'),
            'formatted_citations': {
                'apa': apa_citation,
                'bibtex': bibtex
            }
        }

        return result

    except Exception as e:
        return {
            'error': True,
            'message': str(e),
            'identifier': identifier
        }


def cleanup():
    """Clean up OpenAlex client resources."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
