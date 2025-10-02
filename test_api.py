import os
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize client
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Simple test
def test_simple_prompt():
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What is 15 * 847? Think step by step."}
        ]
    )
    
    # Print response
    print("Response:")
    for block in message.content:
        if block.type == "text":
            print(block.text)
    
    print(f"\nTokens used: {message.usage}")

if __name__ == "__main__":
    test_simple_prompt()