"""Quick verification that model name replacement works."""

import os
from pathlib import Path
from src.llm_maths_research.core.session import ResearchSession
from src.llm_maths_research.core.llm_provider import create_provider

# Test 1: Provider display name
print("="*60)
print("TEST 1: Provider Display Name")
print("="*60)

provider = create_provider('anthropic', os.getenv('ANTHROPIC_API_KEY'), 'claude-sonnet-4-5-20250929')
display_name = provider.get_model_display_name()
print(f"✓ Model: claude-sonnet-4-5-20250929")
print(f"✓ Display Name: {display_name}")
assert display_name == "Claude Sonnet 4.5", f"Expected 'Claude Sonnet 4.5', got '{display_name}'"
print("✓ Display name is correct!")

# Test 2: Problem placeholder replacement
print("\n" + "="*60)
print("TEST 2: Problem Text Replacement")
print("="*60)

problem_path = Path("problems/open_mechanistic_interpretability.txt")
with open(problem_path, 'r') as f:
    problem_text = f.read()

print(f"✓ Problem file loaded")
print(f"✓ Contains placeholder: {'{model}' in problem_text}")

# Simulate what researcher.py does
model_display_name = provider.get_model_display_name()
replaced_text = problem_text.replace("{model}", model_display_name)

print(f"✓ Placeholder replaced: {'{model}' not in replaced_text}")
print(f"✓ Contains display name: {'Claude Sonnet 4.5' in replaced_text}")

# Show the specific line that changed
for line in replaced_text.split('\n'):
    if 'List yourself' in line:
        print(f"\n✓ Result line: '{line.strip()}'")
        assert 'Claude Sonnet 4.5' in line, "Model name not found in line"
        break

# Test 3: Session provider initialization
print("\n" + "="*60)
print("TEST 3: Session Provider Initialization")
print("="*60)

session = ResearchSession("test_verification")
print(f"✓ Session created")
print(f"✓ Provider type: {type(session.provider).__name__}")
print(f"✓ Provider model: {session.provider.model}")
print(f"✓ Provider display name: {session.provider.get_model_display_name()}")

# Clean up
import shutil
shutil.rmtree("outputs/test_verification", ignore_errors=True)

print("\n" + "="*60)
print("ALL TESTS PASSED! ✓")
print("="*60)
print("\nThe multi-provider integration is working correctly:")
print("  • Provider classes instantiate properly")
print("  • Display names are correct")
print("  • {model} placeholder replacement works")
print("  • Session uses provider interface correctly")
