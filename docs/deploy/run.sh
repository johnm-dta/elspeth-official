#!/bin/bash
# run.sh - Elspeth launcher for operators
# ═══════════════════════════════════════════════════════════════
# DEPLOYMENT CONFIGURATION - Edit these for your environment
# ═══════════════════════════════════════════════════════════════

# Container image location (uncomment ONE):
# ELSPETH_IMAGE="ghcr.io/your-org/elspeth:latest"           # GitHub Container Registry
# ELSPETH_IMAGE="your-registry.azurecr.io/elspeth:latest"   # Azure Container Registry
ELSPETH_IMAGE="ghcr.io/your-org/elspeth:latest"

# Key Vault configuration (leave empty to use local secrets.yaml)
KEY_VAULT_URL=""
KEY_VAULT_SECRET_NAME="elspeth-secrets"

# ═══════════════════════════════════════════════════════════════
# END CONFIGURATION - Do not edit below this line
# ═══════════════════════════════════════════════════════════════

set -e

# Colors for output (disabled if not a terminal)
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    NC='\033[0m' # No Color
else
    GREEN=''
    RED=''
    YELLOW=''
    NC=''
fi

echo "═══════════════════════════════════════"
echo "  Elspeth Pipeline Runner"
echo "═══════════════════════════════════════"
echo ""

# Step 1: Check Docker is available
echo -n "Checking Docker... "
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗${NC}"
    echo "  ERROR: Docker is not installed or not in PATH"
    echo "  Install with: sudo apt install docker.io"
    exit 1
fi
if ! docker info &> /dev/null; then
    echo -e "${RED}✗${NC}"
    echo "  ERROR: Docker daemon is not running or you don't have permission"
    echo "  Try: sudo systemctl start docker"
    echo "  Or add yourself to docker group: sudo usermod -aG docker \$USER"
    exit 1
fi
echo -e "${GREEN}✓${NC}"

# Step 2: Pull image
echo -n "Pulling image... "
if docker pull "$ELSPETH_IMAGE" --quiet > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "  ERROR: Failed to pull image: $ELSPETH_IMAGE"
    echo "  Check: Do you have access to the container registry?"
    exit 1
fi

# Step 3: Resolve secrets
echo -n "Loading secrets... "
if [[ -f "secrets.yaml" ]]; then
    echo -e "${GREEN}✓${NC} (local file)"
elif [[ -n "$KEY_VAULT_URL" ]]; then
    # Extract vault name from URL
    VAULT_NAME=$(echo "$KEY_VAULT_URL" | sed 's|https://||' | sed 's|\.vault\.azure\.net.*||')

    if az keyvault secret show --vault-name "$VAULT_NAME" --name "$KEY_VAULT_SECRET_NAME" \
        --query value -o tsv > secrets.yaml 2>/dev/null; then
        echo -e "${GREEN}✓${NC} (Key Vault)"
        # Ensure secrets file is not world-readable
        chmod 600 secrets.yaml
    else
        echo -e "${RED}✗${NC}"
        echo "  ERROR: Failed to fetch secrets from Key Vault"
        echo "  Vault: $VAULT_NAME"
        echo "  Secret: $KEY_VAULT_SECRET_NAME"
        echo ""
        echo "  Check:"
        echo "  - Is the VM's managed identity configured?"
        echo "  - Does the identity have 'get' permission on secrets?"
        exit 1
    fi
else
    echo -e "${RED}✗${NC}"
    echo "  ERROR: No secrets.yaml found and KEY_VAULT_URL not configured"
    echo ""
    echo "  Options:"
    echo "  1. Create secrets.yaml from secrets.yaml.example"
    echo "  2. Set KEY_VAULT_URL in this script"
    exit 1
fi

# Step 4: Validate config exists
echo -n "Checking config... "
if [[ -f "config.yaml" ]]; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "  ERROR: config.yaml not found"
    echo "  Make sure you're running from the correct directory"
    exit 1
fi

# Step 5: Run pipeline
echo "───────────────────────────────────────"
echo "Running pipeline..."
echo ""

docker run --rm \
    -v "$(pwd):/workspace" \
    "$ELSPETH_IMAGE" \
    --settings config.yaml \
    --secrets secrets.yaml

EXIT_CODE=$?

echo ""
echo "───────────────────────────────────────"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✓ Complete.${NC} Results written to configured output."
else
    echo -e "${RED}✗ Pipeline failed${NC} (exit code: $EXIT_CODE)"
    if [[ -d "diagnostics" ]]; then
        echo ""
        echo "  Diagnostics saved to: diagnostics/"
        echo "  Send this folder to support for investigation."
    fi
fi

exit $EXIT_CODE
