# Environment Variables

The `elspeth` CLI automatically loads environment variables from a `.env` file in the current working directory.

## Automatic .env Loading

When you run `elspeth`, it will:

1. Look for a `.env` file in the current directory
2. Parse and load all `KEY=VALUE` pairs
3. Skip comments (lines starting with `#`) and empty lines
4. Only set variables that aren't already in the environment (environment variables take precedence)

This means you can simply run:

```bash
elspeth --settings example/simple/settings.yaml
```

Instead of:

```bash
OPENROUTER_API_KEY=... OPENROUTER_MODEL=... elspeth --settings example/simple/settings.yaml
```

## .env File Format

Your `.env` file should contain `KEY=VALUE` pairs, one per line:

```bash
# OpenRouter Configuration
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
OPENROUTER_MODEL=openai/gpt-4o-mini

# Optional settings
OPENROUTER_SITE_URL=https://example.com
OPENROUTER_APP_NAME=my-app

# Azure OpenAI (if using Azure)
DMP_AZURE_OPENAI_DEPLOYMENT=gpt-4
DMP_AZURE_OPENAI_API_KEY=your-azure-key
DMP_AZURE_OPENAI_API_VERSION=2024-02-15-preview
DMP_AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

## Environment Variable Precedence

Environment variables set in your shell take precedence over `.env` file values:

```bash
# This will use the shell value, not the .env value
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet elspeth --settings settings.yaml
```

## Using in Configuration Files

Reference environment variables in your YAML configuration files using the `_env` suffix:

```yaml
llm:
  plugin: openrouter
  options:
    config:
      # Loads from OPENROUTER_API_KEY environment variable
      api_key_env: OPENROUTER_API_KEY

      # Loads from OPENROUTER_MODEL environment variable
      model_env: OPENROUTER_MODEL
```

## Security Best Practices

1. **Never commit `.env` files** - They contain secrets
   - The `.env` file is already in `.gitignore`
   - Commit `.env.example` instead with placeholder values

2. **Use environment-specific files** if needed:
   ```bash
   .env              # Local development (gitignored)
   .env.example      # Template (committed)
   .env.production   # Production secrets (gitignored)
   ```

3. **Rotate keys regularly** - Especially API keys

4. **Use different keys per environment** - Don't reuse production keys in development

## Common Environment Variables

### OpenRouter
- `OPENROUTER_API_KEY` - Your OpenRouter API key (required)
- `OPENROUTER_MODEL` - Model to use (default: `openai/gpt-4o-mini`)
- `OPENROUTER_SITE_URL` - Your site URL for rankings (optional)
- `OPENROUTER_APP_NAME` - Your app name for rankings (optional)

### Azure OpenAI
- `DMP_AZURE_OPENAI_DEPLOYMENT` - Deployment name
- `DMP_AZURE_OPENAI_API_KEY` - Azure API key
- `DMP_AZURE_OPENAI_API_VERSION` - API version
- `DMP_AZURE_OPENAI_ENDPOINT` - Azure endpoint URL

### Azure Storage
- Variables used by Azure Blob datasources and sinks
- See `docs/architecture/subsystems.md` for details

## Troubleshooting

### "Missing required config value 'api_key'"

1. Check that `.env` exists in your current directory:
   ```bash
   ls -la .env
   ```

2. Verify the file has the correct format:
   ```bash
   cat .env
   ```

3. Ensure you're running `elspeth` from the directory containing `.env`

4. Check for typos in variable names

### Variables Not Loading

1. Make sure the `.env` file is in the current working directory
2. Check that lines don't have leading/trailing spaces
3. Verify `=` is not surrounded by spaces: `KEY=value` not `KEY = value`
4. Remove quotes unless they're part of the value

### Debug Loading

Run with debug logging to see what's happening:

```bash
elspeth --log-level DEBUG --settings settings.yaml
```

Look for the log message:
```
DEBUG:dmp.cli:Loaded environment variables from .env
```

## Examples

### Basic Setup

```bash
# 1. Create .env file
cat > .env << EOF
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=openai/gpt-4o-mini
EOF

# 2. Run elspeth - .env is automatically loaded
elspeth --settings example/simple/settings.yaml
```

### Override for Testing

```bash
# Use a different model for this run only
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet elspeth --settings settings.yaml
```

### Multiple Environments

```bash
# Development
cp .env.example .env
# Edit .env with dev credentials

# Production (on server)
cp .env.production .env
# Run with production config
```
