# Pipeline Runner

## Prerequisites

- Docker installed (`sudo apt install docker.io`)
- Access to Azure (managed identity or local credentials)

## Quick Start

1. **Configure secrets** (first time only):
   ```bash
   cp secrets.yaml.example secrets.yaml
   # Edit secrets.yaml with your values
   ```

2. **Run the pipeline**:
   ```bash
   ./run.sh
   ```

3. **Check results**:
   - Results are written to the configured output location
   - Check the console for progress and any errors

## Troubleshooting

If the pipeline fails:

1. Check the console output for error messages
2. Look in the `diagnostics/` folder for detailed error information
3. Send the `diagnostics/` folder to support if you need help

## What This Pipeline Does

[Describe what your specific pipeline does here]

## Support

Contact: [your-support-contact]
