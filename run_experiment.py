import argparse
from datetime import datetime
from src.llm_maths_research import ScaffoldedResearcher

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run mathematical research experiment')
    parser.add_argument('--papers', nargs='+', required=True, help='ArXiv paper IDs to use (e.g., 2501.00123 2501.00456)')
    parser.add_argument('--max-iterations', type=int, default=2, help='Maximum number of iterations (default: 2)')
    parser.add_argument('--session-name', type=str, help='Custom session name (default: auto-generated)')
    args = parser.parse_args()

    with open('problems/open_research.txt', 'r') as f:
        problem = f.read()

    session_name = args.session_name or f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    researcher = ScaffoldedResearcher(
        session_name=session_name,
        max_iterations=args.max_iterations,
        paper_ids=args.papers
    )

    researcher.run(problem)
