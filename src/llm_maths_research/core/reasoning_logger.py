"""Logger for capturing and storing reasoning/thinking content from LLM providers."""

import json
import os
from datetime import datetime
from typing import Optional


class ReasoningLogger:
    """Handles logging of reasoning content to both JSONL and Markdown formats."""

    def __init__(self, output_dir: str):
        """
        Initialize the reasoning logger.

        Args:
            output_dir: Directory where reasoning logs will be stored (typically session logs dir)
        """
        self.output_dir = output_dir
        self.jsonl_path = os.path.join(output_dir, "reasoning.jsonl")
        self.markdown_path = os.path.join(output_dir, "reasoning_readable.md")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Initialize markdown file with header if it doesn't exist
        if not os.path.exists(self.markdown_path):
            with open(self.markdown_path, 'w') as f:
                f.write("# Reasoning Log\n\n")
                f.write("This file contains human-readable reasoning content from LLM calls throughout the session.\n\n")
                f.write("---\n\n")

    def log_reasoning(
        self,
        iteration: int,
        stage: str,
        model: str,
        reasoning_content: Optional[str],
        reasoning_tokens: Optional[int] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> None:
        """
        Log reasoning content to both JSONL and Markdown files.

        Args:
            iteration: Iteration number (1-indexed)
            stage: Stage name ("planning", "coding", "writing")
            model: Model name/identifier
            reasoning_content: The actual reasoning/thinking text (None if not available)
            reasoning_tokens: Number of reasoning tokens used (if available)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        # Skip if no reasoning content
        if not reasoning_content:
            return

        timestamp = datetime.now().isoformat()

        # Write to JSONL
        jsonl_entry = {
            "iteration": iteration,
            "stage": stage,
            "timestamp": timestamp,
            "model": model,
            "reasoning": reasoning_content,
        }

        # Add token counts if available
        if reasoning_tokens is not None:
            jsonl_entry["reasoning_tokens"] = reasoning_tokens
        if input_tokens is not None:
            jsonl_entry["input_tokens"] = input_tokens
        if output_tokens is not None:
            jsonl_entry["output_tokens"] = output_tokens

        with open(self.jsonl_path, 'a') as f:
            f.write(json.dumps(jsonl_entry) + '\n')

        # Write to Markdown
        with open(self.markdown_path, 'a') as f:
            f.write(f"## Iteration {iteration}\n\n")
            f.write(f"### {stage.title()} Stage\n\n")
            f.write(f"**Model:** {model}  \n")

            # Build token info line
            token_parts = []
            if reasoning_tokens is not None:
                token_parts.append(f"{reasoning_tokens:,} reasoning")
            if input_tokens is not None:
                token_parts.append(f"{input_tokens:,} input")
            if output_tokens is not None:
                token_parts.append(f"{output_tokens:,} output")

            if token_parts:
                f.write(f"**Tokens:** {' | '.join(token_parts)}  \n")

            f.write(f"**Timestamp:** {timestamp}  \n\n")
            f.write(f"{reasoning_content}\n\n")
            f.write("---\n\n")

    def get_jsonl_path(self) -> str:
        """Get the path to the JSONL reasoning log."""
        return self.jsonl_path

    def get_markdown_path(self) -> str:
        """Get the path to the Markdown reasoning log."""
        return self.markdown_path
