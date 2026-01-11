#!/usr/bin/env python3
"""Test script to validate the example configuration without making API calls."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env
env_file = project_root / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value

from elspeth.config import load_settings


def test_configuration():
    """Test that the configuration loads correctly."""
    print("Testing Simple OpenRouter Example Configuration")
    print("=" * 50)
    print()

    # Check environment variables
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model = os.getenv("OPENROUTER_MODEL", "")

    print("✓ Environment Variables:")
    if api_key and api_key != "your_openrouter_api_key_here":
        print(f"  - OPENROUTER_API_KEY: {api_key[:20]}..." if len(api_key) > 20 else f"  - OPENROUTER_API_KEY: [hidden]")
    else:
        print("  ✗ OPENROUTER_API_KEY: Not set or using placeholder")
        print("    Please update your .env file with a valid API key")
        return False

    print(f"  - OPENROUTER_MODEL: {model or 'openai/gpt-4o-mini (default)'}")
    print()

    # Load configuration
    try:
        settings_path = project_root / "example" / "simple" / "settings.yaml"
        settings = load_settings(settings_path, profile="default")
        print("✓ Configuration loaded successfully")
        print()

        # Check data source
        print("✓ Data Source:")
        print(f"  - Type: {type(settings.datasource).__name__}")
        print(f"  - Path: data/sample_input.csv")

        # Check if input file exists
        input_path = project_root / "example" / "simple" / "data" / "sample_input.csv"
        if input_path.exists():
            print(f"  - Input file exists: ✓")
            with open(input_path) as f:
                lines = f.readlines()
                print(f"  - Rows: {len(lines) - 1} (excluding header)")
        else:
            print(f"  - Input file exists: ✗")
        print()

        # Check LLM
        print("✓ LLM Configuration:")
        print(f"  - Type: {type(settings.llm).__name__}")
        print(f"  - Model: {settings.llm.model}")
        print(f"  - Temperature: {settings.llm.temperature}")
        print(f"  - Max Tokens: {settings.llm.max_tokens}")
        print()

        # Check sinks
        print("✓ Output Configuration:")
        for i, sink in enumerate(settings.sinks, 1):
            print(f"  - Sink {i}: {type(sink).__name__}")
        print(f"  - Output path: output/results.csv")
        print()

        # Check prompts
        print("✓ Prompts:")
        system_preview = settings.orchestrator_config.llm_prompt.get("system", "")[:80]
        user_preview = settings.orchestrator_config.llm_prompt.get("user", "")[:80]
        print(f"  - System prompt: {system_preview}...")
        print(f"  - User prompt: {user_preview}...")
        print(f"  - Prompt fields: {settings.orchestrator_config.prompt_fields}")
        print()

        # Check rate limiting
        if settings.rate_limiter:
            print("✓ Rate Limiting:")
            print(f"  - Type: {type(settings.rate_limiter).__name__}")
            print()

        print("=" * 50)
        print("✅ All configuration checks passed!")
        print()
        print("To run the pipeline:")
        print("  cd", project_root)
        print("  ./example/simple/run.sh")
        print()
        print("Or:")
        print("  source .venv/bin/activate")
        print("  elspeth --settings example/simple/settings.yaml")
        print()
        return True

    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_configuration()
    sys.exit(0 if success else 1)
