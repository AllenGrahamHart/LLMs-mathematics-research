import os
from anthropic import Anthropic
from dotenv import load_dotenv
from io import StringIO
import contextlib
from datetime import datetime

# Load API key
load_dotenv()
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Create output directory
os.makedirs("outputs", exist_ok=True)

def extract_code_blocks(text: str):
    """Extract Python code blocks from markdown"""
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

def execute_code(code: str):
    """Execute Python code and capture output"""
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    try:
        with contextlib.redirect_stdout(stdout_capture), \
             contextlib.redirect_stderr(stderr_capture):
            exec(code, {})
        
        return {
            'success': True,
            'stdout': stdout_capture.getvalue(),
            'stderr': stderr_capture.getvalue()
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': stdout_capture.getvalue(),
            'stderr': stderr_capture.getvalue() + f"\n{type(e).__name__}: {str(e)}"
        }

# Generate timestamp for filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"outputs/execution_log_{timestamp}.txt"

# Improved prompt
prompt = """Write Python code that:
1. Prints 'Hello, World!'
2. Prints the numbers 1 through 5

CRITICAL INSTRUCTIONS:
- Provide ONLY executable Python code
- No explanations, no markdown, no comments outside the code
- The code should run directly when executed
- Start immediately with the code
"""

# Open file for writing
with open(output_file, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("TEST: ASK CLAUDE TO WRITE CODE, THEN EXECUTE IT\n")
    f.write("=" * 60 + "\n")
    
    # Step 1: Ask Claude to write code
    f.write("\n1. Asking Claude to write code...\n")
    f.write(f"Prompt: {prompt}\n")
    
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract response text
    response_text = ""
    for block in response.content:
        if block.type == "text":
            response_text += block.text
    
    f.write("\n2. Claude's response:\n")
    f.write(response_text + "\n")
    
    # Step 2: Extract code blocks
    code_blocks = extract_code_blocks(response_text)
    f.write(f"\n3. Found {len(code_blocks)} code block(s)\n")
    
    # Step 3: Execute each code block
    for i, code in enumerate(code_blocks):
        f.write(f"\n4. Executing code block {i + 1}:\n")
        f.write("--- CODE ---\n")
        f.write(code + "\n")
        f.write("--- EXECUTION ---\n")
        
        result = execute_code(code)
        
        if result['success']:
            f.write("✓ Execution successful!\n")
            f.write("Output:\n")
            f.write(result['stdout'] + "\n")
        else:
            f.write("✗ Execution failed!\n")
            f.write("Error:\n")
            f.write(result['stderr'] + "\n")

print(f"Results saved to: {output_file}")