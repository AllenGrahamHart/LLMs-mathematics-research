"""Abstract LLM provider interface and concrete implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Iterator
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    stop_reason: Optional[str] = None
    model: Optional[str] = None
    reasoning_content: Optional[str] = None  # Chain of thought/thinking content (if exposed by provider)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def create_message(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
        system: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        """Create a message and return a standardized response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system: System prompt (optional)
            thinking_budget: Extended thinking budget tokens (optional, provider-specific)
            extra_headers: Additional headers (optional, provider-specific)

        Returns:
            LLMResponse with standardized fields
        """
        pass

    @abstractmethod
    def create_message_stream(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
        system: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Iterator[str]:
        """Create a streaming message.

        Args:
            Same as create_message

        Yields:
            Text chunks from the streaming response
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens
        """
        pass

    @abstractmethod
    def get_model_display_name(self) -> str:
        """Get human-readable model name for use in papers/prompts.

        Returns:
            Display name like "Claude Sonnet 4.5", "GPT-4", etc.
        """
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.anthropic = anthropic
        print(f"[DEBUG] AnthropicProvider initialized: model={model}")

    def create_message(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
        system: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        """Create an Anthropic message using streaming (required for long operations)."""
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if system:
            kwargs["system"] = system

        if thinking_budget:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        # Use streaming to avoid 10-minute timeout on long operations
        content = ""
        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                content += text

            # Get final message with usage stats
            final_message = stream.get_final_message()

        # Extract reasoning content from ThinkingBlocks (if any)
        reasoning_content = ""
        for block in final_message.content:
            if block.type == "thinking":
                reasoning_content += block.thinking

        return LLMResponse(
            content=content,
            input_tokens=final_message.usage.input_tokens,
            output_tokens=final_message.usage.output_tokens,
            cache_creation_tokens=getattr(final_message.usage, 'cache_creation_input_tokens', 0),
            cache_read_tokens=getattr(final_message.usage, 'cache_read_input_tokens', 0),
            stop_reason=final_message.stop_reason,
            model=final_message.model,
            reasoning_content=reasoning_content if reasoning_content else None,
        )

    def create_message_stream(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
        system: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Iterator[str]:
        """Create a streaming Anthropic message."""
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if system:
            kwargs["system"] = system

        if thinking_budget:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        if extra_headers:
            kwargs["extra_headers"] = extra_headers

        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    def count_tokens(self, text: str) -> int:
        """Count tokens using Anthropic's tokenizer."""
        response = self.client.messages.count_tokens(
            model=self.model,
            messages=[{"role": "user", "content": text}]
        )
        return response.input_tokens

    def get_model_display_name(self) -> str:
        """Get display name for Claude models."""
        # Map model IDs to display names
        model_map = {
            "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
            "claude-sonnet-4-20250514": "Claude Sonnet 4",
            "claude-opus-4-20250514": "Claude Opus 4",
            "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
            "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
            "claude-3-5-sonnet-20240620": "Claude 3.5 Sonnet",
        }
        return model_map.get(self.model, f"Claude ({self.model})")


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, api_key: str, model: str, reasoning_effort: str = 'medium'):
        super().__init__(api_key, model)
        from openai import OpenAI
        self.reasoning_effort = reasoning_effort
        print(f"[DEBUG] OpenAIProvider initialized: model={model}, reasoning_effort={reasoning_effort}")
        self.client = OpenAI(
            api_key=api_key,
            timeout=None  # No timeout - let thinking models take as long as needed
        )

    def create_message(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
        system: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        """Create an OpenAI message (non-streaming for compatibility)."""
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages, system)

        # GPT-5 models use max_completion_tokens instead of max_tokens
        # Check if this is a GPT-5 model
        uses_completion_tokens = 'gpt-5' in self.model.lower() or 'o1' in self.model.lower()

        # Build the API call parameters (non-streaming to avoid organization verification requirement)
        api_params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
        }

        # Use the appropriate parameter name based on the model
        if uses_completion_tokens:
            api_params["max_completion_tokens"] = max_tokens
        else:
            api_params["max_tokens"] = max_tokens

        # Add reasoning_effort for GPT-5 and o1 models
        if uses_completion_tokens and self.reasoning_effort:
            api_params["reasoning_effort"] = self.reasoning_effort

        # Use non-streaming mode (streaming requires organization verification for GPT-5)
        response = self.client.chat.completions.create(**api_params)

        # Extract content
        content = response.choices[0].message.content or ""

        # Extract usage stats
        if response.usage:
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            # Extract cache tokens if available (for o1 and GPT-5 models)
            if hasattr(usage, 'prompt_tokens_details'):
                cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
            else:
                cached_tokens = 0
        else:
            input_tokens = 0
            output_tokens = 0
            cached_tokens = 0

        # Get stop reason and model
        stop_reason = response.choices[0].finish_reason if response.choices else None
        model_name = response.model if hasattr(response, 'model') else self.model

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cached_tokens,
            stop_reason=stop_reason,
            model=model_name,
        )

    def create_message_stream(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
        system: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Iterator[str]:
        """Create a streaming OpenAI message."""
        openai_messages = self._convert_messages(messages, system)

        # GPT-5 models use max_completion_tokens instead of max_tokens
        uses_completion_tokens = 'gpt-5' in self.model.lower() or 'o1' in self.model.lower()

        # Build the API call parameters
        api_params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
            "stream": True,
        }

        # Use the appropriate parameter name based on the model
        if uses_completion_tokens:
            api_params["max_completion_tokens"] = max_tokens
        else:
            api_params["max_tokens"] = max_tokens

        stream = self.client.chat.completions.create(**api_params)

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _convert_messages(self, messages: List[Dict[str, Any]], system: Optional[str] = None) -> List[Dict[str, str]]:
        """Convert Anthropic-style messages to OpenAI format."""
        openai_messages = []

        # Add system message if provided
        if system:
            openai_messages.append({"role": "system", "content": system})

        # Convert messages, stripping Anthropic-specific features
        for msg in messages:
            content = msg.get("content", "")

            # Handle list-style content (with cache_control, etc.)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts)

            openai_messages.append({
                "role": msg.get("role", "user"),
                "content": content
            })

        return openai_messages

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        import tiktoken

        # Get encoding for model
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Default to cl100k_base for newer models
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))

    def get_model_display_name(self) -> str:
        """Get display name for GPT models."""
        model_map = {
            "chatgpt-5": "GPT-5",
            "gpt-4": "GPT-4",
            "gpt-4-turbo": "GPT-4 Turbo",
            "gpt-4o": "GPT-4o",
            "gpt-4o-mini": "GPT-4o Mini",
            "o1": "o1",
            "o1-mini": "o1-mini",
            "o1-preview": "o1-preview",
        }
        return model_map.get(self.model, f"GPT ({self.model})")


class GoogleProvider(LLMProvider):
    """Google Gemini provider."""

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        import google.generativeai as genai
        print(f"[DEBUG] GoogleProvider initialized: model={model}")
        # Configure with timeout for long-running requests
        genai.configure(
            api_key=api_key,
            transport='rest',  # Use REST transport for better timeout control
        )
        self.client = genai.GenerativeModel(model)
        self.genai = genai

    def create_message(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
        system: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        """Create a Gemini message using streaming to avoid timeouts."""
        # Convert messages to Gemini format
        gemini_messages = self._convert_messages(messages)

        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        # Note: Google Gemini does not support thinking_config parameter in the API
        # The thinking_budget in config is kept for documentation purposes only
        # Gemini does not expose reasoning content via the API

        # Add system instruction if provided
        if system:
            # Recreate client with system instruction
            self.client = self.genai.GenerativeModel(
                self.model,
                system_instruction=system
            )

        # Use streaming to avoid hanging on slow/unresponsive API
        response_stream = self.client.generate_content(
            gemini_messages,
            generation_config=generation_config,
            stream=True,
        )

        # Accumulate content from stream
        # Note: Google Gemini does not expose reasoning/thought content via API
        content = ""
        final_chunk = None
        for chunk in response_stream:
            if chunk.text:
                content += chunk.text
            final_chunk = chunk

        # Get exact token counts and cache info from final chunk
        if final_chunk and hasattr(final_chunk, 'usage_metadata'):
            input_tokens = final_chunk.usage_metadata.prompt_token_count
            output_tokens = final_chunk.usage_metadata.candidates_token_count
            cached_tokens = getattr(final_chunk.usage_metadata, 'cached_content_token_count', 0)
        else:
            # Fallback if usage not available
            input_tokens = 0
            output_tokens = 0
            cached_tokens = 0

        # Get stop reason from final chunk
        stop_reason = None
        if final_chunk and hasattr(final_chunk, 'candidates') and final_chunk.candidates:
            stop_reason = str(final_chunk.candidates[0].finish_reason)

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cached_tokens,
            stop_reason=stop_reason,
            model=self.model,
            reasoning_content=None,  # Google does not expose reasoning content
        )

    def create_message_stream(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
        system: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Iterator[str]:
        """Create a streaming Gemini message."""
        gemini_messages = self._convert_messages(messages)

        generation_config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }

        if system:
            self.client = self.genai.GenerativeModel(
                self.model,
                system_instruction=system
            )

        response = self.client.generate_content(
            gemini_messages,
            generation_config=generation_config,
            stream=True,
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Convert Anthropic-style messages to Gemini format."""
        gemini_messages = []

        for msg in messages:
            content = msg.get("content", "")

            # Handle list-style content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts)

            # Map roles (Gemini uses 'user' and 'model' instead of 'assistant')
            role = msg.get("role", "user")
            if role == "assistant":
                role = "model"

            gemini_messages.append({
                "role": role,
                "parts": [content]
            })

        return gemini_messages

    def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's tokenizer."""
        result = self.client.count_tokens(text)
        return result.total_tokens

    def get_model_display_name(self) -> str:
        """Get display name for Gemini models."""
        model_map = {
            "gemini-2.5-pro": "Gemini 2.5 Pro",
            "gemini-2.5-flash": "Gemini 2.5 Flash",
            "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
            "gemini-2.0-flash-exp": "Gemini 2.0 Flash",
            "gemini-2.0-flash": "Gemini 2.0 Flash",
            "gemini-exp-1206": "Gemini Experimental 1206",
            "gemini-2.0-flash-thinking-exp-01-21": "Gemini 2.0 Flash Thinking",
            "gemini-1.5-pro": "Gemini 1.5 Pro",
            "gemini-1.5-flash": "Gemini 1.5 Flash",
        }
        return model_map.get(self.model, f"Gemini ({self.model})")


class xAIProvider(LLMProvider):
    """xAI Grok provider."""

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        from openai import OpenAI
        print(f"[DEBUG] xAIProvider initialized: endpoint=https://api.x.ai/v1, model={model}")
        # xAI uses OpenAI-compatible API
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            timeout=None  # No timeout - let thinking models take as long as needed
        )

    def create_message(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
        system: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        """Create a Grok message using streaming to avoid timeouts."""
        # Convert messages to OpenAI format
        openai_messages = self._convert_messages(messages, system)

        # Use streaming to avoid hanging on slow/unresponsive API
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True}  # Request usage stats in stream
        )

        # Note: xAI Grok does not expose reasoning content via OpenAI-compatible API
        content = ""
        final_chunk = None

        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    content += delta.content
            final_chunk = chunk

        # Extract usage stats from final chunk
        if final_chunk and hasattr(final_chunk, 'usage') and final_chunk.usage:
            usage = final_chunk.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            # Extract cache tokens if available (Grok-4 automatic caching)
            if hasattr(usage, 'prompt_tokens_details'):
                cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
            else:
                cached_tokens = 0
        else:
            # Fallback if usage not available in stream
            input_tokens = 0
            output_tokens = 0
            cached_tokens = 0

        # Get stop reason and model from final chunk
        stop_reason = None
        model_name = self.model
        if final_chunk and final_chunk.choices and len(final_chunk.choices) > 0:
            stop_reason = final_chunk.choices[0].finish_reason
        if final_chunk and hasattr(final_chunk, 'model'):
            model_name = final_chunk.model

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cached_tokens,
            stop_reason=stop_reason,
            model=model_name,
            reasoning_content=None,  # xAI does not expose reasoning content
        )

    def create_message_stream(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
        system: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Iterator[str]:
        """Create a streaming Grok message."""
        openai_messages = self._convert_messages(messages, system)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _convert_messages(self, messages: List[Dict[str, Any]], system: Optional[str] = None) -> List[Dict[str, str]]:
        """Convert Anthropic-style messages to OpenAI format."""
        openai_messages = []

        if system:
            openai_messages.append({"role": "system", "content": system})

        for msg in messages:
            content = msg.get("content", "")

            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts)

            openai_messages.append({
                "role": msg.get("role", "user"),
                "content": content
            })

        return openai_messages

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (approximate for Grok)."""
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def get_model_display_name(self) -> str:
        """Get display name for Grok models."""
        model_map = {
            "grok-4-0709": "Grok 4",
            "grok-beta": "Grok Beta",
            "grok-vision-beta": "Grok Vision Beta",
        }
        return model_map.get(self.model, f"Grok ({self.model})")


class MoonshotProvider(LLMProvider):
    """Moonshot Kimi provider."""

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        from openai import OpenAI
        # Kimi uses OpenAI-compatible API (international endpoint)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.ai/v1",
            timeout=None  # No timeout - let thinking models take as long as needed
        )
        print(f"[DEBUG] MoonshotProvider initialized: endpoint=https://api.moonshot.ai/v1, model={model}")

    def create_message(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
        system: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        """Create a Kimi message using streaming to avoid timeouts."""
        openai_messages = self._convert_messages(messages, system)

        # Use streaming to avoid hanging on slow/unresponsive API
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True}  # Request usage stats in stream
        )

        content = ""
        reasoning_content = ""
        final_chunk = None

        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                # Capture regular content
                if delta.content:
                    content += delta.content
                # Capture reasoning content (Kimi K2 exposes this)
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
            # Keep the last chunk which should have usage stats
            final_chunk = chunk

        # Extract usage stats from final chunk
        if final_chunk and hasattr(final_chunk, 'usage') and final_chunk.usage:
            usage = final_chunk.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            # Extract cache tokens (Kimi K2 automatic caching uses same fields as Anthropic)
            cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0)
            cache_read_tokens = getattr(usage, 'cache_read_input_tokens', 0)
        else:
            # Fallback if usage not available in stream
            input_tokens = 0
            output_tokens = 0
            cache_creation_tokens = 0
            cache_read_tokens = 0

        # Get stop reason and model from final chunk
        stop_reason = None
        model_name = self.model
        if final_chunk and final_chunk.choices and len(final_chunk.choices) > 0:
            stop_reason = final_chunk.choices[0].finish_reason
        if final_chunk and hasattr(final_chunk, 'model'):
            model_name = final_chunk.model

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            stop_reason=stop_reason,
            model=model_name,
            reasoning_content=reasoning_content if reasoning_content else None,
        )

    def create_message_stream(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 1.0,
        system: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Iterator[str]:
        """Create a streaming Kimi message."""
        openai_messages = self._convert_messages(messages, system)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _convert_messages(self, messages: List[Dict[str, Any]], system: Optional[str] = None) -> List[Dict[str, str]]:
        """Convert Anthropic-style messages to OpenAI format."""
        openai_messages = []

        if system:
            openai_messages.append({"role": "system", "content": system})

        for msg in messages:
            content = msg.get("content", "")

            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts)

            openai_messages.append({
                "role": msg.get("role", "user"),
                "content": content
            })

        return openai_messages

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (approximate for Kimi)."""
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def get_model_display_name(self) -> str:
        """Get display name for Kimi models."""
        model_map = {
            "kimi-k2-thinking": "Kimi K2 Thinking",
            "moonshot-v1-8k": "Kimi (8k)",
            "moonshot-v1-32k": "Kimi (32k)",
            "moonshot-v1-128k": "Kimi (128k)",
        }
        return model_map.get(self.model, f"Kimi ({self.model})")


def create_provider(provider_name: str, api_key: str, model: str, **provider_kwargs) -> LLMProvider:
    """Factory function to create LLM providers.

    Args:
        provider_name: Name of provider ('anthropic', 'openai', 'google', 'xai', 'moonshot')
        api_key: API key for the provider
        model: Model ID to use
        **provider_kwargs: Provider-specific keyword arguments (e.g., reasoning_effort for OpenAI)

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If provider_name is not supported
    """
    providers = {
        'anthropic': AnthropicProvider,
        'openai': OpenAIProvider,
        'google': GoogleProvider,
        'xai': xAIProvider,
        'moonshot': MoonshotProvider,
    }

    provider_class = providers.get(provider_name.lower())
    if not provider_class:
        raise ValueError(
            f"Unsupported provider: {provider_name}. "
            f"Supported providers: {', '.join(providers.keys())}"
        )

    # Only pass provider_kwargs to providers that support them
    if provider_name.lower() == 'openai':
        return provider_class(api_key, model, **provider_kwargs)
    else:
        return provider_class(api_key, model)
