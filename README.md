# elspeth-simple

**Secure, pluggable orchestration for responsible data transformation and decision-making**

elspeth-simple is a lightweight Python framework for building reliable, auditable data processing workflows. Designed for **Official: Sensitive environments**, it provides production-grade orchestration with appropriate security controls without the overhead of more security-focused products designed for higher classification levels.

## Why elspeth-simple?

Modern data workflows require flexible transformation, analysis, and decision-making systems. Whether using LLMs, traditional algorithms, or custom logic, elspeth-simple provides:

- **Structured orchestration** following the Sense/Decide/Act pattern
- **Pluggable architecture** supporting multiple decision systems (LLMs, rule engines, custom transforms), data sources, and outputs
- **Production features** including retry logic, checkpointing, rate limiting, and cost tracking
- **Appropriate security** with HMAC signing, input validation, and security-level metadata
- **A/B testing support** for comparing baseline and variant approaches
- **Configuration-first design** with hierarchical YAML-based settings

Whether you're processing CSV data with LLM providers like GPT-4, applying rule-based transformations, running statistical analysis, or building custom data pipelines, elspeth-simple provides the scaffolding you need.

## The Sense/Decide/Act Philosophy

elspeth-simple organizes workflows around a three-phase pattern:

1. **SENSE** - Load and prepare input data from datasources (CSV, Azure Blob, custom sources)
2. **DECIDE** - Process each row through decision-making systems (LLMs, rule engines, algorithms) and transformation plugins to make decisions, extract insights, or transform data
3. **ACT** - Output results to multiple sinks (CSV, Excel, blob storage, Git repositories)

This pattern creates clear, auditable workflows where each phase has defined responsibilities and boundaries. The DECIDE phase is intentionally flexible - use LLMs for natural language tasks, traditional algorithms for structured logic, or custom transformations for domain-specific processing.

## Running as Code (Development)

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/johnm-dta/elspeth-simple.git
cd elspeth-simple

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Add your API keys to `.env`:
```
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
```

### Run Your First Workflow

```bash
# Run the simple example
elspeth --settings example/simple/settings.yaml

# Preview first 10 results
elspeth --settings example/simple/settings.yaml --head 10

# Save results to CSV
elspeth --settings example/simple/settings.yaml --output-csv results.csv
```

## Building the Container

The container is fully self-contained with all dependencies (including `scipy` for statistical analysis and `zip` for archive splitting).

### Prerequisites

Install Docker and BuildKit:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y docker.io docker-buildx

# Add yourself to docker group (optional, avoids sudo)
sudo usermod -aG docker $USER
# Log out and back in for group change to take effect

# Verify installation
docker --version
docker buildx version
```

For other platforms, see [Docker installation docs](https://docs.docker.com/get-docker/).

### Build the Image

```bash
# Build with BuildKit (recommended)
docker buildx build -t elspeth:latest .

# Or with legacy builder
docker build -t elspeth:latest .
```

## Running as a Container

The container expects your configuration and data to be mounted at `/workspace`.

### Basic Usage

```bash
docker run --rm \
  -v "$(pwd)":/workspace \
  --env-file .env \
  elspeth:latest \
  --settings /workspace/config/settings.yaml \
  --output-csv /workspace/output/results.csv
```

### With Azure Credentials

```bash
docker run --rm \
  -v "$(pwd)":/workspace \
  -e AZURE_OPENAI_API_KEY \
  -e AZURE_OPENAI_ENDPOINT \
  elspeth:latest \
  --settings /workspace/settings.yaml
```

### Example: Run the Simple Example

```bash
docker run --rm \
  -v "$(pwd)":/workspace \
  --env-file .env \
  elspeth:latest \
  --settings /workspace/example/simple/settings.yaml \
  --head 10
```

### Container Details

- **Base image**: `python:3.13-slim`
- **Working directory**: `/workspace` (mount your files here)
- **User**: Runs as non-root `appuser` (UID 1000)
- **Entrypoint**: `elspeth`
- **Included tools**: `zip` (for archive splitting), `scipy` (for statistics)

## Configuration Overview

elspeth-simple uses **hierarchical YAML configuration** with clear precedence rules:

```yaml
default:
  # Data source configuration
  datasource:
    plugin: local_csv
    options:
      path: data/input.csv

  # LLM configuration
  llm:
    plugin: openrouter
    options:
      api_key: ${OPENROUTER_API_KEY}
      model: ${OPENROUTER_MODEL}

  # Prompt templates (Jinja2)
  prompts:
    system: "You are a data analyst."
    user: "Analyze this record: {field1}, {field2}"

  # Fields to include in prompts
  prompt_fields:
    - field1
    - field2

  # Output sinks
  sinks:
    - plugin: csv
      options:
        path: output/results.csv

  # Optional: Transform plugins
  row_plugins:
    - plugin: custom_transform
      options:
        param: value

  # Optional: Rate limiting
  rate_limiter:
    plugin: adaptive
    options:
      tokens_per_second: 1000

  # Optional: Cost tracking
  cost_tracker:
    plugin: fixed_price
    options:
      prompt_token_price: 0.00015
      completion_token_price: 0.0006

  # Concurrency settings
  concurrency:
    max_workers: 4
    batch_size: 10
```

### Configuration Hierarchy

Configurations merge with **clear precedence** (higher numbers override lower):

1. **System defaults** (precedence=1)
2. **Prompt packs** (precedence=2)
3. **Profile configuration** (precedence=3)
4. **Suite defaults** (precedence=4)
5. **Experiment-level config** (precedence=5)

Use `--explain-config KEY` to see where any configuration value comes from:

```bash
elspeth --settings config.yaml --explain-config llm.options.model
```

## Key Features

### 🔌 Plugin System

Everything is pluggable:

- **Datasources**: CSV, Azure Blob, or build custom loaders
- **LLM Clients**: OpenRouter, Azure OpenAI, or mock for testing
- **Sinks**: CSV, Excel, blob storage, GitHub repos, ZIP bundles
- **Transforms**: Custom row-level and aggregation transforms
- **Rate Limiters**: Fixed window, adaptive token-aware, or none
- **Cost Trackers**: Track spending across experiments

### 🔒 Security Features

- **HMAC Signing**: Cryptographically sign artifacts with SHA-256/SHA-512
- **Security Levels**: Tag data with classification levels
- **Input Validation**: JSON Schema-based configuration validation
- **Secure Defaults**: Timing-safe signature verification

### 🧪 A/B Testing

Compare baseline and variant approaches:

```yaml
# Use ExperimentalOrchestrator
suite:
  orchestrator: experimental

cycles:
  - name: baseline
    metadata:
      is_baseline: true
    llm:
      options:
        model: gpt-4o-mini

  - name: variant-a
    llm:
      options:
        model: gpt-4o
    baseline_plugins:
      - plugin: comparison
        options:
          metric: accuracy
```

### ⚡ Production Features

- **Checkpointing**: Resume interrupted processing
- **Retry Logic**: Configurable backoff and max attempts
- **Rate Limiting**: Respect API limits with adaptive throttling
- **Cost Tracking**: Monitor token usage and spending
- **Concurrency**: Multi-threaded row processing
- **Early Stopping**: Halt conditions for budget/quality gates

### 📊 Flexible Prompting

Jinja2-based templates with field substitution:

```yaml
prompts:
  system: |
    You are an expert classifier.
    Classification criteria:
    {% for criterion in criteria %}
    - {{ criterion }}
    {% endfor %}

  user: |
    Classify this item:
    Name: {{ name }}
    Description: {{ description }}
    Category: {{ category|default('unknown') }}

prompt_fields:
  - name
  - description
  - category

prompt_aliases:
  item_name: name
```

## CLI Reference

```bash
# Basic usage
elspeth --settings CONFIG [OPTIONS]

# Common options
--profile PROFILE           # Select configuration profile
--print-config             # Display resolved config and exit
--explain-config KEY       # Show config value origin
--suite-root DIR           # Override suite directory
--single-run               # Force single run (not suite)
--live-outputs             # Enable actual sink writes
--head N                   # Preview N rows
--output-csv PATH          # Save results to CSV
--disable-metrics          # Remove metrics plugins

# Examples
elspeth --settings config.yaml --profile production
elspeth --settings config.yaml --explain-config llm.plugin
elspeth --settings config.yaml --head 20 --output-csv preview.csv
```

## Project Structure

```
elspeth-simple/
├── src/elspeth/
│   ├── core/              # Core orchestration
│   │   ├── sda/          # SDA runner and suite logic
│   │   ├── prompts/      # Prompt compilation
│   │   ├── security/     # HMAC signing
│   │   └── validation.py # Config validation
│   ├── plugins/          # Plugin implementations
│   │   ├── datasources/  # Data loaders
│   │   ├── llms/         # LLM clients
│   │   ├── sinks/        # Output handlers
│   │   ├── rate_limiters/
│   │   └── cost_trackers/
│   ├── orchestrators/    # Suite orchestration
│   └── cli.py           # Command-line interface
├── tests/               # Test suite
├── example/            # Example configurations
│   ├── simple/         # Basic CSV workflow
│   ├── complex/        # Advanced multi-plugin
│   ├── experimental/   # A/B testing example
│   └── meta-plugins/   # v3.0 meta-plugin patterns
└── docs/               # Documentation

```

## Examples

### Simple CSV Analysis

```yaml
# settings.yaml
default:
  datasource:
    plugin: local_csv
    options:
      path: reviews.csv

  llm:
    plugin: openrouter
    options:
      api_key: ${OPENROUTER_API_KEY}
      model: openai/gpt-4o-mini

  prompts:
    system: "You are a sentiment analyzer."
    user: "Rate the sentiment of: {review_text}"

  prompt_fields: [review_text]

  sinks:
    - plugin: csv
      options:
        path: output/sentiment.csv
```

### Azure Blob to Excel

```yaml
default:
  datasource:
    plugin: azure_blob
    options:
      account_name: myaccount
      container: input-data
      blob_path: data.csv
      profile: production

  llm:
    plugin: azure_openai
    options:
      deployment: gpt-4
      api_key: ${AZURE_OPENAI_KEY}
      endpoint: ${AZURE_OPENAI_ENDPOINT}

  sinks:
    - plugin: excel
      options:
        path: output/analysis.xlsx
        sheet_name: Results
```

## Documentation

- **[Plugin Reference](docs/plugins/)** - Comprehensive plugin documentation
  - Meta-plugins (FieldCollector, FieldExpander)
  - Data reshape plugins
  - Metrics analyzers
- **[Schema-Driven Execution](docs/schema-driven-execution.md)** - v3.0 architecture
- **[Migration Guide](docs/migration-v3.0-metrics.md)** - v2.x to v3.0 upgrade path
- **[CLAUDE.md](CLAUDE.md)** - Developer guide for Claude Code
- **[Configuration Precedence](docs/configuration-precedence.md)** - Config merging rules

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/core/test_config_merger.py -v
```

### Code Quality

```bash
# Linting
ruff check src/ tests/

# Type checking
mypy src/

# Format check
ruff format --check src/ tests/
```

## When to Use elspeth-simple vs. Other Products

**Use elspeth-simple when:**
- Working with official-sensitive data or lower classifications
- You need production features without heavy security overhead
- Rapid iteration and experimentation are priorities
- Your threat model doesn't require supply chain verification, SBOM generation, or advanced container security

**Consider more security-focused products when:**
- Handling SECRET or TOP SECRET data
- Requiring comprehensive CI/CD security gates (Semgrep, Bandit, Gitleaks, Grype)
- Need automated dependency auditing and security updates
- Supply chain security and provenance tracking are critical
- Container signing and verification are required

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure linting and type checking pass
5. Submit a pull request

## Support

- **Issues**: [GitHub Issues](https://github.com/johnm-dta/elspeth-simple/issues)
- **Documentation**: See `/docs` directory
- **Examples**: See `/example` directory

---

Built with a focus on **clarity, security, and extensibility** for intelligent data workflows.
