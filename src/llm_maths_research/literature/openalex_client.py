"""
OpenAlex API client for literature search and retrieval.

This module provides a client for interacting with the OpenAlex API,
which indexes over 240 million scholarly works.
"""

import time
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import requests


class OpenAlexClient:
    """Client for OpenAlex API interactions."""

    BASE_URL = "https://api.openalex.org"
    DEFAULT_RATE_LIMIT_DELAY = 0.1  # 100ms between requests (polite API usage)

    def __init__(self, email: Optional[str] = None, cache_dir: Optional[Path] = None):
        """
        Initialize OpenAlex client.

        Args:
            email: Email for polite API usage (recommended but optional)
            cache_dir: Directory for caching API responses (optional)
        """
        self.email = email
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.last_request_time = 0

        # Set up session parameters
        if email:
            self.session.params = {'mailto': email}

        # Initialize cache
        self._cache = {}
        self.cache_file = None
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = cache_dir / "openalex_cache.json"
            self._load_cache()

    def _load_cache(self):
        """Load cache from disk if it exists."""
        if self.cache_file and self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}

    def _save_cache(self):
        """Save cache to disk."""
        if self.cache_file:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self._cache, f, indent=2)
            except IOError:
                pass  # Fail silently on cache write errors

    def _rate_limit(self):
        """Implement rate limiting to be polite to OpenAlex API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.DEFAULT_RATE_LIMIT_DELAY:
            time.sleep(self.DEFAULT_RATE_LIMIT_DELAY - time_since_last)

        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None,
                     use_cache: bool = True) -> Dict[str, Any]:
        """
        Make a request to OpenAlex API with caching and error handling.

        Args:
            endpoint: API endpoint (e.g., '/works' or '/works/W123')
            params: Query parameters
            use_cache: Whether to use cached results

        Returns:
            JSON response as dictionary

        Raises:
            requests.exceptions.RequestException: On API errors
        """
        # Build cache key
        cache_key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"

        # Check cache
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Rate limit
        self._rate_limit()

        # Make request
        url = f"{self.BASE_URL}{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Cache result
            if use_cache:
                self._cache[cache_key] = data
                self._save_cache()

            return data

        except requests.exceptions.Timeout:
            raise Exception(f"Request to OpenAlex timed out: {url}")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limited - wait and retry once
                time.sleep(2)
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            else:
                raise Exception(f"HTTP error from OpenAlex: {e.response.status_code} - {e.response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error connecting to OpenAlex: {str(e)}")

    def search_works(self, query: Optional[str] = None,
                    filters: Optional[Dict[str, Any]] = None,
                    per_page: int = 10, page: int = 1,
                    sort: str = "relevance_score:desc") -> Dict[str, Any]:
        """
        Search for works in OpenAlex.

        Args:
            query: Search query string
            filters: Dictionary of filters to apply
            per_page: Results per page (1-200)
            page: Page number
            sort: Sort order (e.g., "cited_by_count:desc", "publication_date:desc")

        Returns:
            Dictionary with 'meta' and 'results' keys
        """
        params = {
            'per_page': min(per_page, 200),
            'page': page,
            'sort': sort
        }

        # Add search query
        if query:
            params['search'] = query

        # Build filter string
        if filters:
            filter_parts = []

            # Citation network filters
            if 'cites' in filters:
                filter_parts.append(f"cites:{filters['cites']}")
            if 'cited_by' in filters:
                filter_parts.append(f"cited_by:{filters['cited_by']}")

            # Temporal filters
            if 'publication_year' in filters:
                filter_parts.append(f"publication_year:{filters['publication_year']}")
            elif 'from_year' in filters or 'to_year' in filters:
                from_year = filters.get('from_year', 1900)
                to_year = filters.get('to_year', 2100)
                filter_parts.append(f"publication_year:{from_year}-{to_year}")

            # Quality filters
            if 'min_citations' in filters:
                filter_parts.append(f"cited_by_count:>{filters['min_citations']}")

            # Content filters
            if 'is_open_access' in filters:
                filter_parts.append(f"is_oa:{str(filters['is_open_access']).lower()}")
            if 'publication_type' in filters:
                filter_parts.append(f"type:{filters['publication_type']}")
            if 'has_fulltext' in filters:
                filter_parts.append(f"has_fulltext:{str(filters['has_fulltext']).lower()}")

            # Identifier filters (for verification)
            if 'doi' in filters:
                filter_parts.append(f"doi:{filters['doi']}")
            if 'title' in filters:
                # Use title.search for fuzzy matching
                params['filter'] = f"title.search:{filters['title']}"
            if 'authors' in filters:
                params['filter'] = f"authorships.author.display_name.search:{filters['authors']}"

            # Combine filters
            if filter_parts:
                if 'filter' in params:
                    params['filter'] += ',' + ','.join(filter_parts)
                else:
                    params['filter'] = ','.join(filter_parts)

        # Override sort if specified in filters
        if filters and 'sort_by' in filters:
            sort_field = filters['sort_by']
            params['sort'] = f"{sort_field}:desc"

        return self._make_request('/works', params)

    def get_work(self, work_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a single work.

        Args:
            work_id: OpenAlex ID (W123), DOI, or full OpenAlex URL

        Returns:
            Dictionary with work details
        """
        # Clean up work_id
        if work_id.startswith('https://openalex.org/'):
            work_id = work_id.split('/')[-1]
        elif work_id.startswith('https://doi.org/'):
            work_id = 'https://doi.org/' + work_id.split('doi.org/')[-1]
        elif work_id.startswith('10.'):
            work_id = 'https://doi.org/' + work_id

        return self._make_request(f'/works/{work_id}')

    def get_abstract(self, work_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract plain text abstract from work data.

        OpenAlex stores abstracts as inverted indices for legal reasons.
        This method reconstructs the plain text.

        Args:
            work_data: Work dictionary from API

        Returns:
            Plain text abstract or None if not available
        """
        abstract_inverted = work_data.get('abstract_inverted_index')
        if not abstract_inverted:
            return None

        # Reconstruct abstract from inverted index
        # Format: {"word": [position1, position2, ...]}
        word_positions = []
        for word, positions in abstract_inverted.items():
            for pos in positions:
                word_positions.append((pos, word))

        # Sort by position and join
        word_positions.sort(key=lambda x: x[0])
        abstract = ' '.join(word for _, word in word_positions)

        return abstract

    def close(self):
        """Clean up resources."""
        self.session.close()
        self._save_cache()
