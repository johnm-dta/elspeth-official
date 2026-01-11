# OpenRouter Setup Guide

OpenRouter provides access to multiple LLM providers through a unified API, making it easy to switch between models from OpenAI, Anthropic, Google, and others.

## Quick Start

### 1. Get an OpenRouter API Key

1. Visit [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Sign up or log in
3. Create a new API key
4. Add credits to your account at [https://openrouter.ai/credits](https://openrouter.ai/credits)

### 2. Configure Environment Variables

Edit the `.env` file in your project root:

```bash
# Required: Your OpenRouter API key
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here

# Optional: Specify the model (defaults to openai/gpt-4o-mini)
OPENROUTER_MODEL=openai/gpt-4o-mini

# Optional: Your site URL for OpenRouter rankings
OPENROUTER_SITE_URL=https://github.com/yourusername/project

# Optional: Your app name for OpenRouter rankings
OPENROUTER_APP_NAME=elspeth-simple
```

### 3. Configure Your Settings File

Create a YAML configuration file (e.g., `settings.yaml`):

```yaml
llm:
  type: openrouter
  config:
    api_key_env: OPENROUTER_API_KEY
    model_env: OPENROUTER_MODEL
    temperature: 0.7
    max_tokens: 2000

datasource:
  type: local_csv
  path: data/input.csv

sinks:
  - type: csv
    path: output/results.csv
    overwrite: true
```

### 4. Run Your Workflow

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run elspeth with your configuration
elspeth --settings settings.yaml
```

## Available Models

OpenRouter provides access to many models. Here are some popular options:

### Cost-Effective Options
- `openai/gpt-4o-mini` - Fast and affordable GPT-4 model (default)
- `google/gemini-flash-1.5` - Google's efficient model
- `meta-llama/llama-3.1-8b-instruct` - Open-source Llama model

### High-Performance Options
- `openai/gpt-4o` - Latest GPT-4 model
- `anthropic/claude-3.5-sonnet` - Anthropic's most capable model
- `google/gemini-pro-1.5` - Google's flagship model

### Specialized Options
- `openai/gpt-4-vision` - For image understanding
- `perplexity/llama-3.1-sonar-large-128k-online` - With web search capabilities

See the full list at: [https://openrouter.ai/models](https://openrouter.ai/models)

## Configuration Options

The OpenRouter client supports the following configuration options:

```yaml
llm:
  type: openrouter
  config:
    # API Key (required)
    api_key: "your-key-here"           # Direct key
    api_key_env: OPENROUTER_API_KEY    # Or from environment

    # Model Selection (optional, defaults to openai/gpt-4o-mini)
    model: "openai/gpt-4o-mini"        # Direct model name
    model_env: OPENROUTER_MODEL        # Or from environment

    # Generation Parameters (optional)
    temperature: 0.7                    # 0.0 to 2.0
    max_tokens: 2000                    # Maximum response length

    # OpenRouter-specific (optional)
    site_url: "https://example.com"     # For rankings
    app_name: "My App"                  # For rankings
```

## Using Environment Variables

The OpenRouter client automatically loads configuration from environment variables:

1. **API Key**: Set via `api_key_env` (points to env var name)
2. **Model**: Set via `model_env` (points to env var name) or defaults to `OPENROUTER_MODEL`
3. **Fallback**: If `OPENROUTER_MODEL` is not set, uses `openai/gpt-4o-mini`

## Cost Tracking

OpenRouter charges based on the model you use. Enable cost tracking in your configuration:

```yaml
cost_tracking:
  enabled: true
  warn_threshold: 10.0  # Warn when costs exceed $10
```

View your usage and costs at: [https://openrouter.ai/activity](https://openrouter.ai/activity)

## Rate Limiting

Add rate limiting to avoid hitting API limits:

```yaml
rate_limit:
  calls_per_minute: 10
  calls_per_hour: 100
```

## Troubleshooting

### "OpenRouterClient missing required config value 'api_key'"

- Ensure your `.env` file has `OPENROUTER_API_KEY` set
- Check that the environment variable is loaded (run `echo $OPENROUTER_API_KEY`)
- Verify your config file has `api_key_env: OPENROUTER_API_KEY`

### "Authentication failed" or 401 errors

- Verify your API key is correct
- Ensure you have credits in your OpenRouter account
- Check that your key hasn't expired

### "Model not found" errors

- Verify the model name is correct (check [https://openrouter.ai/models](https://openrouter.ai/models))
- Some models require special access or higher credit limits

## Example: Complete Configuration

See `docs/openrouter_example.yaml` for a complete working example.

## Comparing Costs

OpenRouter shows costs for each model in USD per million tokens. For reference:

- `openai/gpt-4o-mini`: ~$0.15/1M input tokens, ~$0.60/1M output tokens
- `anthropic/claude-3.5-sonnet`: ~$3/1M input tokens, ~$15/1M output tokens
- `google/gemini-flash-1.5`: ~$0.075/1M input tokens, ~$0.30/1M output tokens

Check current pricing at: [https://openrouter.ai/models](https://openrouter.ai/models)
