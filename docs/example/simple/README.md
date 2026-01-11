# Simple OpenRouter Example

This example demonstrates a basic SDA (Sense/Decide/Act) pipeline that:
1. **SENSE**: Reads customer feedback from a CSV file
2. **DECIDE**: Processes each row with OpenRouter LLM to analyze sentiment
3. **ACT**: Writes the results to an output CSV file

## Quick Start

### 1. Set up your API key

Copy the `.env.example` file to the project root and add your OpenRouter API key:

```bash
# From the project root directory
cp example/simple/.env.example .env
# Edit .env and add your actual API key
```

Or if you already have a `.env` file in the project root with `OPENROUTER_API_KEY`, you're all set!

### 2. Run the example

```bash
# Make sure you're in the project root directory

# Run the pipeline - .env is automatically loaded!
.venv/bin/elspeth --settings example/simple/settings.yaml
```

The `elspeth` CLI automatically loads environment variables from `.env` in the current directory, so you don't need to manually export them.

### 3. Check the results

The processed results will be written to `example/simple/output/results.csv`

## What's in this example?

### Input Data (`data/sample_input.csv`)

Sample customer feedback with 5 rows:
```csv
id,text,category
1,"This product is absolutely amazing! The quality exceeded my expectations...",positive
2,"Very disappointed. The item broke after just one week of use...",negative
...
```

### Configuration (`settings.yaml`)

The settings file configures the entire pipeline:

- **Data Source**: Reads from `data/sample_input.csv`
- **LLM**: OpenRouter client with:
  - Model: Configured via `OPENROUTER_MODEL` env var
  - Temperature: 0.7
  - Max tokens: 500
- **Prompts**: System and user prompts for sentiment analysis
- **Output**: Results written to `output/results.csv`
- **Rate Limiting**: Max 10 requests per minute

### Output (`output/results.csv`)

The output CSV will contain all the original columns plus:
- `response`: The raw LLM response
- Any additional fields extracted by transform plugins

## Customizing the Example

### Change the Model

Edit your `.env` file to use a different model:
```bash
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

See available models at: https://openrouter.ai/models

### Modify the Prompts

Edit the `prompts` section in `settings.yaml`:

```yaml
prompts:
  system: |
    Your custom system prompt here...

  user: |
    Your custom user prompt here...
    Use {column_name} to reference CSV columns
```

### Change the Input/Output

Edit the datasource and sinks in `settings.yaml`:

```yaml
datasource:
  plugin: local_csv
  options:
    path: path/to/your/input.csv

sinks:
  - plugin: csv
    options:
      path: path/to/your/output.csv
      overwrite: true
```

### Add Transform Plugins

Add transform plugins to extract structured data from LLM responses:

```yaml
row_plugins:
  - name: json_extract
    options:
      field: response
      schema:
        type: object
        properties:
          sentiment:
            type: string
          summary:
            type: string
```

## Understanding the Pipeline

### 1. SENSE Phase (Data Source)

The pipeline reads data from `data/sample_input.csv` using the `CSVDataSource`:

```yaml
datasource:
  plugin: local_csv
  options:
    path: data/sample_input.csv
```

### 2. DECIDE Phase (LLM + Transforms)

For each row, the pipeline:
1. Extracts the specified `prompt_fields` (text, category)
2. Renders the user prompt template with these fields
3. Sends the prompt to OpenRouter
4. Receives the LLM response

The prompts are configured with:
```yaml
prompts:
  system: "You are a helpful assistant..."
  user: "Analyze this: {text}"

prompt_fields:
  - text
  - category
```

### 3. ACT Phase (Result Sink)

Results are written to `output/results.csv`:

```yaml
sinks:
  - plugin: csv
    options:
      path: output/results.csv
      overwrite: true
```

## Rate Limiting

The example uses adaptive rate limiting:

```yaml
rate_limiter:
  plugin: adaptive
  options:
    requests_per_minute: 10
```

This ensures you don't exceed API rate limits and helps manage costs.

## Troubleshooting

### "OpenRouterClient missing required config value 'api_key'"

Make sure:
1. You have a `.env` file in the project root
2. It contains: `OPENROUTER_API_KEY=your_actual_key`
3. The key is valid (test at https://openrouter.ai)

### "No such file or directory: data/sample_input.csv"

Make sure you're running the command from the project root directory:
```bash
cd /path/to/elspeth-simple
elspeth --settings example/simple/settings.yaml
```

### OpenRouter API Errors

- **401 Unauthorized**: Check your API key is correct
- **402 Payment Required**: Add credits at https://openrouter.ai/credits
- **429 Too Many Requests**: Reduce `requests_per_minute` in rate_limiter

## Next Steps

1. Try different models by changing `OPENROUTER_MODEL`
2. Modify the prompts to suit your use case
3. Add your own CSV data
4. Explore other examples in the `example/` directory
5. Read the full documentation in `docs/`

## Cost Considerations

OpenRouter charges per token. The example uses `gpt-4o-mini` by default, which is cost-effective:
- Input: ~$0.15 per 1M tokens
- Output: ~$0.60 per 1M tokens

For 5 rows with ~100 tokens each:
- Total cost: < $0.01

Monitor your usage at: https://openrouter.ai/activity
