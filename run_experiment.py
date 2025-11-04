import argparse
import sys
from datetime import datetime
from pathlib import Path
from src.llm_maths_research import ScaffoldedResearcher

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run mathematical research experiment')
    parser.add_argument('--papers', nargs='+', required=True, help='Paper file names (without .txt extension) from problems/papers/ directory (e.g., Paolo MyPaper)')
    parser.add_argument('--problem', type=str, default='problems/open_research.txt', help='Path to problem file (default: problems/open_research.txt)')
    parser.add_argument('--data', nargs='*', help='Data file names from data/datasets/ directory (e.g., dataset1.csv timeseries/data.json)')
    parser.add_argument('--code', nargs='*', help='Code context names from problems/code/ directory (e.g., nanogpt transformer-xl)')
    parser.add_argument('--max-iterations', type=int, default=2, help='Maximum number of iterations (default: 2)')
    parser.add_argument('--session-name', type=str, help='Custom session name (default: auto-generated)')
    parser.add_argument('--start-iteration', type=int, default=1, help='Starting iteration number for resuming (default: 1)')
    parser.add_argument('--resume-at-critic', type=int, help='Resume at critic phase of this iteration (generator already completed)')
    args = parser.parse_args()

    # Read problem file with error handling
    problem_path = Path(args.problem)
    if not problem_path.exists():
        print(f"Error: Problem file '{args.problem}' not found")
        sys.exit(1)

    with open(problem_path, 'r') as f:
        problem = f.read()

    # Validate paper files exist
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

    # Generate session name with paper ID prefix for clarity
    if args.session_name:
        session_name = args.session_name
    else:
        paper_prefix = "_".join(args.papers) if len(args.papers) <= 2 else args.papers[0]
        session_name = f"{paper_prefix}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    researcher = ScaffoldedResearcher(
        session_name=session_name,
        max_iterations=args.max_iterations,
        paper_ids=args.papers,
        data_ids=args.data,
        code_ids=args.code,
        start_iteration=args.start_iteration,
        resume_at_critic=args.resume_at_critic
    )

    researcher.run(problem)
