# Elspeth Pipeline Runner - Deployment Files

This directory contains files for deploying elspeth pipelines to operator VMs.

## Contents

- `run.sh` - Main launcher script
- `README.md` - This file

## Setting Up an Ops Repo

Copy these files to your deployment repository along with:

1. `config.yaml` - Your pipeline configuration (without secrets)
2. `secrets.yaml.example` - Template showing required secrets
3. `.gitignore` - To exclude secrets and diagnostics

### Example .gitignore

```
secrets.yaml
diagnostics/
```

### Example secrets.yaml.example

```yaml
# Copy this to secrets.yaml and fill in values
# DO NOT commit secrets.yaml to git!

AZURE_STORAGE_KEY: "your-storage-account-key"
OPENAI_API_KEY: "sk-your-api-key"
```

## Operator Instructions

1. Clone the ops repo to the VM
2. If using local secrets:
   - Copy `secrets.yaml.example` to `secrets.yaml`
   - Fill in the secret values
3. Run: `./run.sh`
4. If errors occur, send `diagnostics/` folder to support

## Configuration

Edit the top section of `run.sh` to configure:

- `ELSPETH_IMAGE` - Container registry URL
- `KEY_VAULT_URL` - (Optional) Azure Key Vault for secrets
- `KEY_VAULT_SECRET_NAME` - Name of secret in Key Vault
