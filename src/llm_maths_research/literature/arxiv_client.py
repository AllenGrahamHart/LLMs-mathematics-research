"""
ArXiv client for downloading and extracting paper content.

Provides functions to download ArXiv papers and extract their LaTeX source
with minimal preprocessing (strip comments, extract document body).
"""

import re
import tempfile
import tarfile
from pathlib import Path
from typing import Optional, Dict, Any
import urllib.request
import urllib.error


def extract_arxiv_id_from_url(url: str) -> Optional[str]:
    """
    Extract ArXiv ID from various URL formats.

    Args:
        url: ArXiv URL (e.g., https://arxiv.org/abs/2301.00001)

    Returns:
        ArXiv ID (e.g., "2301.00001") or None
    """
    # Pattern matches: arxiv.org/abs/XXXX.XXXXX or arxiv.org/pdf/XXXX.XXXXX
    match = re.search(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)', url, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def download_arxiv_source(arxiv_id: str, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Download LaTeX source from ArXiv.

    Args:
        arxiv_id: ArXiv ID (e.g., "2301.00001")
        output_dir: Directory to save extracted files (uses temp dir if None)

    Returns:
        Dictionary with 'success', 'tex_file', 'error' keys
    """
    try:
        # Construct source URL
        source_url = f"https://arxiv.org/e-print/{arxiv_id}"

        # Create temp directory if needed
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Download tar.gz file
        tar_path = output_dir / f"{arxiv_id}.tar.gz"

        try:
            urllib.request.urlretrieve(source_url, tar_path)
        except urllib.error.HTTPError as e:
            if e.code == 403:
                return {
                    'success': False,
                    'error': f"ArXiv source not available for {arxiv_id} (may be PDF-only submission)"
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP error {e.code}: {e.reason}"
                }

        # Extract tar.gz
        extract_dir = output_dir / arxiv_id
        extract_dir.mkdir(exist_ok=True)

        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_dir)

        # Find main .tex file (usually the one with \documentclass)
        tex_files = list(extract_dir.glob('*.tex'))

        if not tex_files:
            return {
                'success': False,
                'error': f"No .tex files found in ArXiv source for {arxiv_id}"
            }

        # Find main tex file (contains \documentclass and \begin{document})
        main_tex = None
        for tex_file in tex_files:
            content = tex_file.read_text(encoding='utf-8', errors='ignore')
            if r'\documentclass' in content and r'\begin{document}' in content:
                main_tex = tex_file
                break

        if main_tex is None:
            # Fallback: use first .tex file
            main_tex = tex_files[0]

        return {
            'success': True,
            'tex_file': main_tex,
            'extract_dir': extract_dir
        }

    except Exception as e:
        return {
            'success': False,
            'error': f"Error downloading ArXiv source: {str(e)}"
        }


def extract_document_body(tex_content: str) -> str:
    r"""
    Extract document body from LaTeX source.

    Strips comments and extracts content between \begin{document} and \end{document}.

    Args:
        tex_content: Full LaTeX source code

    Returns:
        Extracted document body
    """
    # Strip comments (but not escaped %)
    tex_content = re.sub(r'(?<!\\)%.*', '', tex_content)

    # Extract document body
    match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', tex_content, re.DOTALL)

    if match:
        body = match.group(1)
        # Clean up multiple blank lines
        body = re.sub(r'\n{3,}', '\n\n', body)
        return body.strip()
    else:
        # Fallback: return everything (some papers might have non-standard structure)
        return tex_content


def get_arxiv_paper(arxiv_id: str, cache_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Download and extract ArXiv paper content.

    Args:
        arxiv_id: ArXiv ID (e.g., "2301.00001")
        cache_dir: Directory for caching downloaded papers

    Returns:
        Dictionary with paper content and metadata:
            - success: bool
            - arxiv_id: str
            - content: str (LaTeX document body)
            - error: str (if success=False)
    """
    # Download source
    download_result = download_arxiv_source(arxiv_id, output_dir=cache_dir)

    if not download_result['success']:
        return {
            'success': False,
            'arxiv_id': arxiv_id,
            'error': download_result['error']
        }

    try:
        # Read tex file
        tex_file = download_result['tex_file']
        tex_content = tex_file.read_text(encoding='utf-8', errors='ignore')

        # Extract document body
        body = extract_document_body(tex_content)

        # Get character and approximate token count
        char_count = len(body)
        approx_tokens = char_count // 4  # Rough estimate: 1 token ≈ 4 chars

        return {
            'success': True,
            'arxiv_id': arxiv_id,
            'content': body,
            'char_count': char_count,
            'approx_tokens': approx_tokens,
            'source_file': str(tex_file)
        }

    except Exception as e:
        return {
            'success': False,
            'arxiv_id': arxiv_id,
            'error': f"Error extracting paper content: {str(e)}"
        }
