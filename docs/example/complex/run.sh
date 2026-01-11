#!/bin/bash
# Runner script for the complex OpenRouter example
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

echo "🚀 Running Complex OpenRouter Example"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This example demonstrates advanced Jinja2 template features:"
echo "  ✓ Conditional logic and sections"
echo "  ✓ String filters and transformations"
echo "  ✓ Multidimensional data handling"
echo "  ✓ Field aliases for clean templates"
echo "  ✓ Complex prompt engineering"
echo ""
echo "Processing 5 products with comprehensive analysis..."
echo ""

# Create output directory if it doesn't exist
mkdir -p output

# Run elspeth from the project root
cd ../..
.venv/bin/elspeth --settings example/complex/settings.yaml

echo ""
echo "✅ Analysis complete!"
echo "📄 Results written to: example/complex/output/analysis.csv"
echo ""
echo "To view results:"
echo "  cat example/complex/output/analysis.csv | head -20"
echo ""
echo "To see a specific product analysis:"
echo "  grep 'P001' example/complex/output/analysis.csv"
