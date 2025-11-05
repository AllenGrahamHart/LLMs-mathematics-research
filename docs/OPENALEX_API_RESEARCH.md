# OpenAlex API Research Summary

## Overview

OpenAlex is a free, open-source alternative to Scopus and Web of Science that indexes over 240 million scholarly works with about 50,000 added daily. The API is completely free with no authentication required.

## Key Capabilities

### 1. Literature Search
- **Full-text search** across titles, abstracts, and (for some works) full text
- **Boolean operators**: AND, OR, NOT (must be uppercase)
- **Field-specific search**: Can search specific fields like `title.search`, `abstract.search`, `fulltext.search`
- **Stemming and stop-word removal** for improved search results
- **Relevance scoring**: Combines text similarity with citation-weighted ranking
- **Autocomplete**: Type-ahead search function for work titles with author hints

### 2. Citation Verification & Analysis
- **Citation counts**: Total citations and normalized percentiles
- **Citation networks**:
  - `cites:W123` - Find works that cite a specific work
  - `cited_by:W123` - Find works cited by a specific work
- **Referenced works**: Outgoing citations from a paper
- **Citation trends**: `counts_by_year` shows citations per year for the last decade
- **Impact metrics**: Field-Weighted Citation Impact (FWCI)

### 3. Metadata Retrieval

Each Work object includes:

**Core Identifiers:**
- OpenAlex ID
- DOI
- MAG ID
- Other identifiers

**Publication Info:**
- Title
- Publication date (exact date and year)
- Publication type (article, preprint, book, etc.)
- Language
- Abstract (as inverted index for legal reasons)

**Author & Institution Data:**
- Up to 100 authors with affiliations
- Corresponding author IDs
- Institution diversity metrics
- Author contributions and affiliations

**Topic & Concept Classification:**
- Related topics with relevance scores
- Academic concepts
- Keywords

**Open Access Information:**
- OA status
- Best OA location
- License information
- Article Processing Charges (APC)

**Citation Data:**
- Total citations
- Citations by year
- Normalized percentiles
- Field-weighted impact

### 4. Filtering Capabilities

Extensive filtering options:

**Temporal Filters:**
- `publication_year:2020`
- `from_publication_date:2024-01-01`
- `to_publication_date:2024-12-31`

**Citation Filters:**
- `cited_by_count:>100`
- `cites:W123` (papers citing W123)
- `cited_by:W456` (papers cited by W456)

**Content Filters:**
- `topics.id:T123`
- `concepts.id:C456`
- `language:en`
- `type:article`
- `is_oa:true`
- `fulltext.search:"pattern formation"`

**Boolean Combinations:**
Filters can be combined with AND/OR logic for complex queries.

## API Technical Details

### Rate Limits
- **100,000 requests per day** per user
- No authentication required (but email recommended)
- No cost

### Endpoints
Base URL: `https://api.openalex.org`

**Main endpoints:**
- `/works` - Search and retrieve scholarly works
- `/authors` - Author information
- `/sources` - Publication venues
- `/institutions` - Research institutions
- `/topics` - Academic topics
- `/concepts` - Academic concepts

### Pagination

**Basic Paging** (for results < 10,000):
```
?page=2&per-page=100
```
- Default: 25 results per page
- Max: 200 results per page
- Limited to first 10,000 results

**Cursor Paging** (for large result sets):
```
?cursor=*
```
- Required for accessing >10,000 results
- Response includes `next_cursor` in meta object
- More complex but no result limit

### Response Format

JSON format with structure:
```json
{
  "meta": {
    "count": 1234,
    "page": 1,
    "per_page": 25,
    "next_cursor": "..."
  },
  "results": [
    {
      "id": "https://openalex.org/W1234567890",
      "doi": "https://doi.org/10.1234/example",
      "title": "Paper Title",
      "publication_year": 2024,
      "publication_date": "2024-03-15",
      "cited_by_count": 42,
      "authorships": [...],
      "topics": [...],
      "referenced_works": [...],
      "open_access": {...},
      ...
    }
  ]
}
```

## Use Cases for LLM Mathematics Research

### 1. Literature Search
- Search for relevant papers by topic/keywords
- Find papers in specific mathematical subfields
- Discover recent work in a research area

### 2. Citation Verification
- Verify if cited papers exist and have correct metadata
- Check citation counts and impact
- Validate author names and affiliations

### 3. Reference Discovery
- Find papers that cite key references
- Build citation networks
- Discover related work through citation links

### 4. Context Building
- Retrieve abstracts and metadata for context
- Identify key papers in a field
- Track research trends over time

### 5. Quality Assessment
- Check citation impact of referenced works
- Verify papers are published in legitimate venues
- Assess field-weighted citation impact

## Implementation Considerations

### For Generator/Researcher Integration:

1. **Search Function**: Allow LLM to search for papers by topic/keywords
   - Input: Search query string
   - Output: List of relevant works with metadata

2. **Citation Lookup**: Verify and retrieve citation details
   - Input: DOI, title, or OpenAlex ID
   - Output: Full work metadata including citations

3. **Related Work Discovery**: Find papers that cite/reference key works
   - Input: OpenAlex Work ID
   - Output: List of citing/cited works

4. **Metadata Extraction**: Pull structured data from papers
   - Input: Work ID or DOI
   - Output: Parsed metadata (authors, year, venue, abstract)

### Recommended Python Approach:

```python
import requests
import time

class OpenAlexClient:
    BASE_URL = "https://api.openalex.org"

    def __init__(self, email=None):
        self.email = email
        self.session = requests.Session()
        if email:
            self.session.params = {'mailto': email}

    def search_works(self, query, filters=None, per_page=25):
        """Search for works by query string"""
        params = {'search': query, 'per_page': per_page}
        if filters:
            params.update(filters)
        return self._get('/works', params)

    def get_work(self, work_id):
        """Get single work by OpenAlex ID or DOI"""
        return self._get(f'/works/{work_id}')

    def get_citations(self, work_id, max_results=100):
        """Get papers that cite this work"""
        params = {'filter': f'cites:{work_id}', 'per_page': max_results}
        return self._get('/works', params)

    def _get(self, endpoint, params=None):
        url = f"{self.BASE_URL}{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        time.sleep(0.1)  # Rate limiting courtesy
        return response.json()
```

## Next Steps

1. Create `src/llm_maths_research/literature/` module
2. Implement OpenAlexClient class with core functionality
3. Add tool functions for LLM to call:
   - `search_literature(query, filters)`
   - `verify_citation(doi_or_title)`
   - `get_related_works(work_id)`
   - `get_citation_context(work_id)`
4. Integrate with researcher prompts
5. Add caching to avoid redundant API calls
6. Handle rate limiting and errors gracefully

## Resources

- Documentation: https://docs.openalex.org/
- API Overview: https://docs.openalex.org/how-to-use-the-api/api-overview
- Works API: https://docs.openalex.org/api-entities/works
- Search Guide: https://docs.openalex.org/api-entities/works/search-works
