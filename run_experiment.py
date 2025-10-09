import argparse
import sys
from datetime import datetime
from pathlib import Path
from src.llm_maths_research import ScaffoldedResearcher

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run mathematical research experiment')
    parser.add_argument('--papers', nargs='+', required=True, help='Paper file names (without .txt extension) from problems/papers/ directory (e.g., Paolo MyPaper)')
    parser.add_argument('--problem', type=str, default='problems/open_research.txt', help='Path to problem file (default: problems/open_research.txt)')
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

    session_name = args.session_name or f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    researcher = ScaffoldedResearcher(
        session_name=session_name,
        max_iterations=args.max_iterations,
        paper_ids=args.papers,
        start_iteration=args.start_iteration,
        resume_at_critic=args.resume_at_critic
    )

    researcher.run(problem)
