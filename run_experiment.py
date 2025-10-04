from scaffolded_researcher import ScaffoldedResearcher
from datetime import datetime

if __name__ == "__main__":
    with open('problems/open_research.txt', 'r') as f:
        problem = f.read()

    researcher = ScaffoldedResearcher(
        session_name=f"open_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        max_iterations=1
    )

    researcher.run(problem)
