import os
from anthropic import Anthropic
from dotenv import load_dotenv
from io import StringIO
import contextlib
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

load_dotenv()
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

class ResearchSession:
    def __init__(self, session_name: str):
        self.session_name = session_name
        self.output_dir = f"outputs/{session_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.latex_file = os.path.join(self.output_dir, "paper.tex")
        self.log_file = os.path.join(self.output_dir, "session_log.txt")
        
        # Initialize empty LaTeX file
        self._initialize_latex()
        
        self.log = []
    
    def _initialize_latex(self):
        """Create initial LaTeX structure"""
        initial_content = r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}

\title{Research Paper}
\author{Claude}
\date{\today}

\begin{document}
\maketitle

% Content will be added here

\end{document}
"""
        with open(self.latex_file, 'w') as f:
            f.write(initial_content)
    
    def extract_code_blocks(self, text: str):
        """Extract Python code blocks"""
        blocks = []
        parts = text.split('```')
        for i in range(1, len(parts), 2):
            block = parts[i]
            if block.startswith('python\n'):
                blocks.append(block[7:])
            elif block.startswith('python '):
                blocks.append(block[7:])
            else:
                blocks.append(block.strip())
        return blocks
    
    def extract_latex_content(self, text: str):
        """Extract LaTeX content from response"""
        # Look for full LaTeX document
        if r'\documentclass' in text:
            # Full document provided
            if '```latex' in text or '```tex' in text:
                parts = text.split('```')
                for i in range(1, len(parts), 2):
                    block = parts[i]
                    if block.startswith('latex\n') or block.startswith('tex\n'):
                        return block.split('\n', 1)[1] if '\n' in block else block
            return text  # Return as-is if no code block
        return None
    
    def execute_code(self, code: str, block_num: int):
        """Execute code and save any plots"""
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        # Namespace with common imports
        namespace = {
            'np': __import__('numpy'),
            'plt': plt,
            'matplotlib': matplotlib,
        }
        
        try:
            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):
                exec(code, namespace)
            
            # Save any matplotlib figures
            figure_paths = []
            for i, fig_num in enumerate(plt.get_fignums()):
                fig = plt.figure(fig_num)
                filename = f"figure_{block_num}_{i}.png"
                path = os.path.join(self.output_dir, filename)
                fig.savefig(path, dpi=150, bbox_inches='tight')
                figure_paths.append(filename)  # Store relative path for LaTeX
                plt.close(fig)
            
            return {
                'success': True,
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue(),
                'figures': figure_paths
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': stdout_capture.getvalue(),
                'stderr': stderr_capture.getvalue() + f"\n{type(e).__name__}: {str(e)}",
                'figures': []
            }
    
    def update_latex(self, new_content: str):
        """Replace LaTeX file with new content"""
        with open(self.latex_file, 'w') as f:
            f.write(new_content)
    
    def write_log(self, entry: str):
        """Append to session log"""
        self.log.append(entry)
        with open(self.log_file, 'a') as f:
            f.write(entry + "\n")
    
    def call_claude(self, prompt: str):
        """Call Claude and process response"""
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text
        
        return response_text
    
    def process_response(self, response: str, iteration: int):
        """Process Claude's response: execute code and/or update LaTeX"""
        self.write_log(f"\n{'='*60}\nITERATION {iteration}\n{'='*60}")
        self.write_log(f"Claude's response:\n{response}\n")
        
        results = {
            'code_executed': False,
            'latex_updated': False,
            'figures_generated': [],
            'execution_results': []
        }
        
        # Check for and execute code blocks
        code_blocks = self.extract_code_blocks(response)
        if code_blocks:
            results['code_executed'] = True
            self.write_log(f"\nFound {len(code_blocks)} code block(s)")
            
            for i, code in enumerate(code_blocks):
                self.write_log(f"\n--- Executing code block {i+1} ---")
                exec_result = self.execute_code(code, iteration)
                results['execution_results'].append(exec_result)
                
                if exec_result['success']:
                    self.write_log(f"✓ Execution successful")
                    if exec_result['stdout']:
                        self.write_log(f"Output:\n{exec_result['stdout']}")
                    if exec_result['figures']:
                        self.write_log(f"Generated figures: {exec_result['figures']}")
                        results['figures_generated'].extend(exec_result['figures'])
                else:
                    self.write_log(f"✗ Execution failed")
                    self.write_log(f"Error:\n{exec_result['stderr']}")
        
        # Check for and update LaTeX
        latex_content = self.extract_latex_content(response)
        if latex_content:
            results['latex_updated'] = True
            self.update_latex(latex_content)
            self.write_log("\n✓ LaTeX file updated")
        
        return results


# Test the system
def test_research_session():
    session = ResearchSession("test_session")
    
    # Test 1: Generate a plot
    print("\n=== TEST 1: Generate a plot ===")
    prompt1 = """Generate a simple plot of sin(x) and cos(x) from 0 to 2π.

Use matplotlib to create the figure. The code should be executable."""
    
    response1 = session.call_claude(prompt1)
    results1 = session.process_response(response1, iteration=1)
    print(f"Code executed: {results1['code_executed']}")
    print(f"Figures generated: {results1['figures_generated']}")
    
    # Test 2: Create initial LaTeX with figure reference
    print("\n=== TEST 2: Create LaTeX document ===")
    prompt2 = f"""Create a LaTeX document with:
- Title: "Test Research Paper"
- A brief introduction
- A section showing the figure generated earlier (filename: {results1['figures_generated'][0] if results1['figures_generated'] else 'figure_1_0.png'})

Provide the COMPLETE LaTeX document including all necessary packages and structure."""
    
    response2 = session.call_claude(prompt2)
    results2 = session.process_response(response2, iteration=2)
    print(f"LaTeX updated: {results2['latex_updated']}")

    # Test 3: Compile the LaTeX document
    print("\n=== TEST 3: Compile LaTeX document ===")
    import subprocess
    latex_dir = session.output_dir
    latex_filename = "paper.tex"

    try:
        # Run pdflatex twice (for references)
        result = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', latex_filename],
            cwd=latex_dir,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            # Run second time for references
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', latex_filename],
                cwd=latex_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            print(f"✓ LaTeX compilation successful")
            print(f"PDF generated: {os.path.join(latex_dir, 'paper.pdf')}")
        else:
            print(f"✗ LaTeX compilation failed")
            print(f"Error output:\n{result.stderr}")
    except FileNotFoundError:
        print("✗ pdflatex not found. Please install a LaTeX distribution (e.g., texlive).")
    except subprocess.TimeoutExpired:
        print("✗ LaTeX compilation timed out")
    except Exception as e:
        print(f"✗ Error during compilation: {e}")

    print(f"\n=== Results saved to: {session.output_dir} ===")
    print(f"LaTeX file: {session.latex_file}")
    print(f"Log file: {session.log_file}")
    print(f"Figures: {results1['figures_generated']}")

if __name__ == "__main__":
    test_research_session()