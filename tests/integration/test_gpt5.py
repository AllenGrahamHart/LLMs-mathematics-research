"""Test GPT-5 API integration with OpenAI."""

import os
import pytest
from openai import OpenAI


@pytest.mark.skipif(
    not os.getenv('OPENAI_API_KEY'),
    reason="OPENAI_API_KEY not set"
)
def test_gpt5_basic_completion():
    """Test basic GPT-5 completion without streaming."""
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key, timeout=None)

    response = client.chat.completions.create(
        model='gpt-5',
        messages=[{'role': 'user', 'content': 'Reply with only the character x and nothing else.'}],
        max_completion_tokens=1,  # GPT-5 uses max_completion_tokens instead of max_tokens
        stream=False  # Don't use streaming (requires org verification)
    )

    content = response.choices[0].message.content
    assert content is not None
    assert 'x' in content.lower()


@pytest.mark.skipif(
    not os.getenv('OPENAI_API_KEY'),
    reason="OPENAI_API_KEY not set"
)
def test_gpt5_reasoning_effort():
    """Test GPT-5 with reasoning_effort parameter."""
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key, timeout=None)

    response = client.chat.completions.create(
        model='gpt-5',
        messages=[{'role': 'user', 'content': 'What is 2+2?'}],
        max_completion_tokens=10,
        reasoning_effort='high',  # Test reasoning_effort parameter
        stream=False
    )

    content = response.choices[0].message.content
    assert content is not None
    assert '4' in content
