from datetime import datetime
from src.llm_maths_research import ScaffoldedResearcher

if __name__ == "__main__":
    with open('problems/open_research_Paolo.txt', 'r') as f:
        problem = f.read()

    researcher = ScaffoldedResearcher(
        session_name=f"open_research_Paolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        max_iterations=4
    )

    researcher.run(problem)
