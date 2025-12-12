"""
Helper script to run evaluation on provided sample files.
Usage: python run_sample_evaluation.py
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cli import cli


def main():
    """Run evaluation on sample files."""
    print("🔍 Looking for sample files...\n")

    # Look for sample files in different locations
    possible_paths = [
        ("../Json/sample-chat-conversation-01.json", "../Json/sample_context_vectors-01.json"),
        (
            "../Json/sample-chat-conversation-02.json",
            "../Json/sample_context_vectors-02.json",
        ),
        ("samples/sample-chat-conversation-01.json", "samples/sample_context_vectors-01.json"),
    ]

    conversation_path = None
    context_path = None

    for conv, ctx in possible_paths:
        if Path(conv).exists() and Path(ctx).exists():
            conversation_path = conv
            context_path = ctx
            break

    if not conversation_path:
        print("❌ Sample files not found!")
        print("\nPlease ensure sample files are in one of these locations:")
        print("  - ../Json/")
        print("  - samples/")
        sys.exit(1)

    print(f"✓ Found sample files:")
    print(f"  Conversation: {conversation_path}")
    print(f"  Context: {context_path}\n")

    # Run CLI with sample files
    sys.argv = [
        "cli",
        "evaluate",
        "--conversation",
        conversation_path,
        "--context",
        context_path,
        "--out",
        "output/sample_results.json",
        "--verbose",
    ]

    cli()


if __name__ == "__main__":
    main()
