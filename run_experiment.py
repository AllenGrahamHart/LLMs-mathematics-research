import argparse
import sys
from datetime import datetime
from pathlib import Path
from src.llm_maths_research import ScaffoldedResearcher
from src.llm_maths_research.config import set_provider
from src.llm_maths_research.provider_defaults import list_providers, get_provider_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run mathematical research experiment with LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Providers:
  anthropic, openai, google, xai, moonshot

  Use --list-providers to see detailed information about each provider.

Examples:
  # Use default provider from config.yaml
  python run_experiment.py --problem problems/open_research.txt

  # Override provider via CLI
  python run_experiment.py --provider openai --problem problems/open_research.txt

  # See provider details
  python run_experiment.py --list-providers
        """
    )
    parser.add_argument('--provider', type=str, choices=list_providers(),
                       help='LLM provider to use (overrides config.yaml). Choices: ' + ', '.join(list_providers()))
    parser.add_argument('--list-providers', action='store_true',
                       help='List available providers with details and exit')
    parser.add_argument('--papers', nargs='*', help='Paper file names (without .txt extension) from problems/papers/ directory (e.g., Paolo MyPaper). Optional - if not provided, researcher will use literature search tools.')
    parser.add_argument('--problem', type=str, default='problems/open_research.txt', help='Path to problem file (default: problems/open_research.txt)')
    parser.add_argument('--data', nargs='*', help='Data file names from data/datasets/ directory (e.g., dataset1.csv timeseries/data.json)')
    parser.add_argument('--code', nargs='*', help='Code context names from problems/code/ directory (e.g., nanogpt transformer-xl)')
    parser.add_argument('--max-iterations', type=int, default=2, help='Maximum number of iterations (default: 2)')
    parser.add_argument('--session-name', type=str, help='Custom session name (default: auto-generated)')
    args = parser.parse_args()

    # Handle --list-providers
    if args.list_providers:
        print("Available LLM Providers")
        print("=" * 80)
        print()
        for provider in list_providers():
            print(get_provider_info(provider))
            print()
        sys.exit(0)

    # Override provider if specified via CLI
    if args.provider:
        print(f"Using provider: {args.provider} (overriding config.yaml)")
        set_provider(args.provider)

    # Read problem file with error handling
    problem_path = Path(args.problem)
    if not problem_path.exists():
        print(f"Error: Problem file '{args.problem}' not found")
        sys.exit(1)

    with open(problem_path, 'r') as f:
        problem = f.read()

    # Validate paper files exist (only if papers provided)
    if args.papers:
        missing_papers = []
        for paper_id in args.papers:
            paper_path = Path(f"problems/papers/{paper_id}.txt")
            if not paper_path.exists():
                missing_papers.append(paper_id)

        if missing_papers:
            print(f"Error: Paper file(s) not found in problems/papers/:")
            for paper_id in missing_papers:
                print(f"  - {paper_id}.txt")
            sys.exit(1)

    # Generate session name
    if args.session_name:
        session_name = args.session_name
    else:
        if args.papers:
            # Use paper IDs as prefix if provided
            paper_prefix = "_".join(args.papers) if len(args.papers) <= 2 else args.papers[0]
            session_name = f"{paper_prefix}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            # Use problem file name as prefix if no papers
            problem_name = problem_path.stem  # Get filename without extension
            session_name = f"{problem_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    researcher = ScaffoldedResearcher(
        session_name=session_name,
        max_iterations=args.max_iterations,
        paper_ids=args.papers,
        data_ids=args.data,
        code_ids=args.code
    )

    researcher.run(problem)
