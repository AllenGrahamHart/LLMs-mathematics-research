"""Command-line interface for LLM Mathematics Research."""

import argparse
from datetime import datetime
from pathlib import Path
from .core.researcher import ScaffoldedResearcher


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Mathematics Research - Automated research with AI"
    )
    parser.add_argument(
        "problem_file",
        type=str,
        help="Path to file containing the research problem"
    )
    parser.add_argument(
        "--session-name",
        type=str,
        default=None,
        help="Custom session name (default: auto-generated from problem file and timestamp)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum number of research iterations (default: 20)"
    )

    args = parser.parse_args()

    # Read problem file
    problem_path = Path(args.problem_file)
    if not problem_path.exists():
        print(f"Error: Problem file '{args.problem_file}' not found")
        return 1

    with open(problem_path, 'r') as f:
        problem = f.read()

    # Generate session name if not provided
    if args.session_name:
        session_name = args.session_name
    else:
        problem_basename = problem_path.stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_name = f"{problem_basename}_{timestamp}"

    # Run research
    researcher = ScaffoldedResearcher(
        session_name=session_name,
        max_iterations=args.max_iterations
    )

    researcher.run(problem)

    return 0


if __name__ == "__main__":
    exit(main())
