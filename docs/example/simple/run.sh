#!/bin/bash
# Simple runner script for the OpenRouter example
#
# Usage: ./run.sh

set -e

# Check if .env exists in project root
if [ ! -f "../../.env" ]; then
    echo "❌ Error: .env file not found in project root"
    echo ""
    echo "Please create a .env file with your OpenRouter API key:"
    echo "  cp example/simple/.env.example .env"
    echo "  # Edit .env and add your API key"
    exit 1
fi

echo "🚀 Running Simple OpenRouter Example"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create output directory if it doesn't exist
mkdir -p output

# Run elspeth from the project root
# Note: elspeth automatically loads .env from the current directory
cd ../..
.venv/bin/elspeth --settings example/simple/settings.yaml

echo ""
echo "✅ Pipeline complete!"
echo "📄 Results written to: example/simple/output/results.csv"
echo ""
echo "To view results:"
echo "  cat example/simple/output/results.csv"
