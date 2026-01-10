# How to Run Elspeth Pipelines

This guide explains how to run elspeth pipelines using Docker with secrets management.

## Prerequisites

- Docker installed and running
- Container image pulled (e.g., `docker pull your-registry/elspeth:latest`)
- Your pipeline configuration files

## Quick Start

### 1. Create Your Secrets File

Create `secrets.yaml` with your sensitive values:

```yaml
# secrets.yaml - DO NOT commit to git!
AZURE_OPENAI_API_KEY: "your-azure-openai-api-key"
AZURE_DEVOPS_PAT: "your-azure-devops-personal-access-token"
AZURE_BLOB_SAS_TOKEN: "sp=rl&st=2025-11-26T10:31:17Z&se=..."
```

### 2. Reference Secrets in Your Config

Use `${VAR_NAME}` syntax in your config to reference secrets:

```yaml
# config.yaml
default:
  datasource:
    plugin: azure_blob
    options:
      config_path: blob_config.yaml
      profile: input

  # ... rest of config
```

```yaml
# blob_config.yaml - storage configuration with secrets
input:
  storage_uri: https://yourstorage.blob.core.windows.net/container/data.csv
  sas_token: "${AZURE_BLOB_SAS_TOKEN}"
```

### 3. Run the Pipeline

```bash
docker run --rm \
  -v "$(pwd):/workspace" \
  your-registry/elspeth:latest \
  --settings config.yaml \
  --secrets secrets.yaml
```

## DMP-RC1 Example (Production Use Case)

This example shows the DMP evaluation pipeline - a real-world use case that:
- Reads candidate data from Azure Blob Storage
- Runs CARES evaluation (10 parallel LLM queries per row)
- Synthesizes feedback (2 summary queries per row)
- Outputs to CSV, creates signed archive bundle, pushes to Azure DevOps

### Directory Structure

```
dmp-evaluation/
├── run.sh                    # Launcher script
├── config.yaml               # Main pipeline config (no secrets)
├── blob_config.yaml          # Azure Blob storage profiles
├── secrets.yaml              # Sensitive values (DO NOT commit)
└── output/                   # Results written here
```

### secrets.yaml

```yaml
# secrets.yaml - DO NOT commit to git!
AZURE_OPENAI_API_KEY: "your-azure-openai-api-key"
AZURE_DEVOPS_PAT: "your-azure-devops-personal-access-token"
INPUT_SAS_TOKEN: "sp=rl&st=2025-11-26T10:31:17Z&se=2025-11-27T18:46:17Z&..."
```

### blob_config.yaml

```yaml
# Azure Blob Storage Configuration
# Separate file for storage profiles - secrets referenced via ${VAR}

input:
  connection_name: dmp_eval_input
  storage_uri: https://dmpevalstorage.blob.core.windows.net/dmp-inputs/candidates.csv
  sas_token: "${INPUT_SAS_TOKEN}"

output:
  connection_name: dmp_eval_output
  storage_uri: https://dmpevalstorage.blob.core.windows.net/dmp-outputs/
  sas_token: "${OUTPUT_SAS_TOKEN}"
```

### config.yaml

```yaml
# DMP Evaluation Pipeline Configuration
# Uses Azure OpenAI with prompt packs from Azure DevOps

default:
  # ============================================================
  # SENSE: Data Source - Read from Azure Blob Storage
  # ============================================================
  datasource:
    plugin: azure_blob
    options:
      config_path: blob_config.yaml
      profile: input

  # ============================================================
  # Checkpointing - allows resuming interrupted batch runs
  # ============================================================
  checkpoint:
    path: checkpoint.jsonl
    field: APPID  # Unique row identifier from source data

  # ============================================================
  # ACT: Output Sinks
  # ============================================================
  sinks:
    # Local CSV for immediate results
    - plugin: csv
      options:
        path: output/results.csv

    # Signed archive bundle for audit trail
    - plugin: azure_archive_bundle
      options:
        base_path: output
        vault_url: "https://your-keyvault.vault.azure.net"
        key_name: archive-signing-key
        archive_name: dmp_evaluation_bundle
        include_patterns:
          - "*.yaml"
          - "*.json"

    # Push to Azure DevOps as immutable record
    - plugin: azure_devops_repo
      security_level: official
      options:
        organization: YourOrganization
        project: YourProject
        repository: DMP
        branch: main
        token_env: AZURE_DEVOPS_PAT
        path_template: "evaluations/{experiment}/{timestamp}"
        commit_message_template: "Add evaluation bundle for {experiment}"
        artifacts:
          consumes:
            - "@archive"
            - "@archive_manifest"
            - "@archive_signature"

  # ============================================================
  # DECIDE: Row Plugins - LLM Query Chains
  # ============================================================
  row_plugins:
    # ----------------------------------------------------------
    # Round 1: CARES Evaluation (10 queries per row)
    # ----------------------------------------------------------
    - plugin: llm_query
      options:
        llm:
          plugin: azure_openai
          options:
            config:
              api_key_env: AZURE_OPENAI_API_KEY
              azure_endpoint: "https://your-openai.openai.azure.com/"
              deployment: "gpt-4o"
              api_version: "2024-02-15-preview"
              temperature: 0.2
              max_tokens: 1000
              response_format: json_object

        # Prompt pack from Azure DevOps (contains system/user prompts)
        prompt_pack: "azuredevops://YourOrg/YourProject/Repo/prompts/candidate-eval"

        middlewares:
          - name: prompt_shield
            options:
              denied_terms:
                - "medicare"
                - "tfn"
                - "tax file number"
              on_violation: log

        retry:
          max_attempts: 3
          initial_delay_seconds: 1.0
          max_delay_seconds: 60.0
          backoff_multiplier: 2.0

        parallel_queries: true
        max_parallel: 10

        rate_limiter:
          plugin: adaptive
          options:
            requests_per_minute: 900

        queries:
          # CAPABILITY Case Study - 5 CARES criteria
          - name: capability_circumstance
            output_key: capability_circumstance
            inputs:
              case_study_text: CS1Summary
              category: CAT
            defaults:
              context: "capability"
              criteria: "Circumstance"

          - name: capability_action
            output_key: capability_action
            inputs:
              case_study_text: CS1Summary
              category: CAT
            defaults:
              context: "capability"
              criteria: "Action"

          - name: capability_result
            output_key: capability_result
            inputs:
              case_study_text: CS1Summary
              category: CAT
            defaults:
              context: "capability"
              criteria: "Result"

          - name: capability_ethics
            output_key: capability_ethics
            inputs:
              case_study_text: CS1Summary
              category: CAT
            defaults:
              context: "capability"
              criteria: "Ethical Behaviour"

          - name: capability_sustainability
            output_key: capability_sustainability
            inputs:
              case_study_text: CS1Summary
              category: CAT
            defaults:
              context: "capability"
              criteria: "Sustainability"

          # CAPACITY Case Study - 5 CARES criteria
          - name: capacity_circumstance
            output_key: capacity_circumstance
            inputs:
              case_study_text: CS2Summary
              category: CAT
            defaults:
              context: "capacity"
              criteria: "Circumstance"

          - name: capacity_action
            output_key: capacity_action
            inputs:
              case_study_text: CS2Summary
              category: CAT
            defaults:
              context: "capacity"
              criteria: "Action"

          - name: capacity_result
            output_key: capacity_result
            inputs:
              case_study_text: CS2Summary
              category: CAT
            defaults:
              context: "capacity"
              criteria: "Result"

          - name: capacity_ethics
            output_key: capacity_ethics
            inputs:
              case_study_text: CS2Summary
              category: CAT
            defaults:
              context: "capacity"
              criteria: "Ethical Behaviour"

          - name: capacity_sustainability
            output_key: capacity_sustainability
            inputs:
              case_study_text: CS2Summary
              category: CAT
            defaults:
              context: "capacity"
              criteria: "Sustainability"

    # ----------------------------------------------------------
    # Round 2: Feedback Synthesis (2 queries per row)
    # Uses outputs from Round 1 via flattened row fields
    # ----------------------------------------------------------
    - plugin: llm_query
      options:
        llm:
          plugin: azure_openai
          options:
            config:
              api_key_env: AZURE_OPENAI_API_KEY
              azure_endpoint: "https://your-openai.openai.azure.com/"
              deployment: "gpt-4o"
              api_version: "2024-02-15-preview"
              temperature: 0.3
              max_tokens: 1000

        prompt_pack: "azuredevops://YourOrg/YourProject/Repo/prompts/summarise"

        retry:
          max_attempts: 3
          initial_delay_seconds: 1.0

        parallel_queries: true
        max_parallel: 10

        rate_limiter:
          plugin: adaptive
          options:
            requests_per_minute: 900

        queries:
          # Capability Summary - uses Round 1 outputs
          - name: capability_summary
            output_key: cs1_summary
            inputs:
              case_study: CS1Summary
              circumstance_score: capability_circumstance_score
              circumstance_rationale: capability_circumstance_rationale
              action_score: capability_action_score
              action_rationale: capability_action_rationale
              result_score: capability_result_score
              result_rationale: capability_result_rationale
              ethics_score: capability_ethics_score
              ethics_rationale: capability_ethics_rationale
              sustainability_score: capability_sustainability_score
              sustainability_rationale: capability_sustainability_rationale

          # Capacity Summary - uses Round 1 outputs
          - name: capacity_summary
            output_key: cs2_summary
            inputs:
              case_study: CS2Summary
              circumstance_score: capacity_circumstance_score
              circumstance_rationale: capacity_circumstance_rationale
              action_score: capacity_action_score
              action_rationale: capacity_action_rationale
              result_score: capacity_result_score
              result_rationale: capacity_result_rationale
              ethics_score: capacity_ethics_score
              ethics_rationale: capacity_ethics_rationale
              sustainability_score: capacity_sustainability_score
              sustainability_rationale: capacity_sustainability_rationale
```

### Running the DMP Pipeline

```bash
# Using the launcher script
./run.sh

# Or manually with Docker
docker run --rm \
  -v "$(pwd):/workspace" \
  your-registry/elspeth:latest \
  --settings config.yaml \
  --secrets secrets.yaml
```

## Using the Launcher Script

For easier operation, use the provided `run.sh` script:

```bash
# Copy from deploy/ops-template/ or deploy/
cp deploy/run.sh .

# Edit the configuration section at the top of run.sh
# to set your container image location

# Run
./run.sh
```

The script will:
- Check Docker is available
- Pull the latest image
- Load secrets (from local file or Azure Key Vault)
- Validate your config exists
- Run the pipeline with progress indicators

## Troubleshooting

### Missing Secrets Error

If you see an error like:
```
Missing secret 'AZURE_OPENAI_API_KEY' - add it to your secrets.yaml file
```

Check that:
1. `secrets.yaml` exists in your current directory
2. The variable name matches exactly (case-sensitive)
3. No typos in `${VAR_NAME}` references in config.yaml

### Pipeline Failures

On error, a `diagnostics/` folder is created with:
- `error-summary.txt` - Human-readable error description
- `config-resolved.yaml` - Your config with secrets redacted
- `stack-trace.txt` - Technical details for support

Send this folder to support for investigation.

### Config Validation

To check your config without running the pipeline:
```bash
docker run --rm \
  -v "$(pwd):/workspace" \
  your-registry/elspeth:latest \
  --settings config.yaml \
  --secrets secrets.yaml \
  --print-config
```

## Security Notes

- Never commit `secrets.yaml` to git
- Add `secrets.yaml` to your `.gitignore`
- Use Azure Key Vault for production deployments
- Diagnostics automatically redact secrets from config dumps
- SAS tokens should have minimal required permissions and short expiry
