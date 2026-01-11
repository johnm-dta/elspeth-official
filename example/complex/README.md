# Complex OpenRouter Example - Advanced Template Features

This example demonstrates the **full power of the Jinja2 template system** with sophisticated prompt engineering for e-commerce product analysis.

## What This Example Shows

### 1. **Conditional Logic**
```jinja2
{%- if notes %}
**Special Notes:** {{ notes }}
{%- endif %}
```
Different sections appear based on data availability.

### 2. **String Filters**
```jinja2
{{ name | title }}      # Capitalize first letter of each word
{{ tier | upper }}      # Convert to uppercase
{{ trend | title }}     # Title case transformation
```

### 3. **Conditional Formatting**
```jinja2
{% if rating >= 4.5 %}Market Leader
{% elif rating >= 4.0 %}Strong Performer
{% elif rating >= 3.5 %}Average Performer
{% else %}Needs Improvement
{% endif %}
```

### 4. **Multidimensional Data Handling**
- Multiple features per product
- Multiple reviews per product
- Multiple competitors per product
- Conditional rendering based on data presence

### 5. **Field Aliases**
```yaml
prompt_aliases:
  product_name: name          # {{ name }} instead of {{ product_name }}
  price_tier: tier            # {{ tier }} instead of {{ price_tier }}
  avg_rating: rating          # {{ rating }} instead of {{ avg_rating }}
```
Makes templates cleaner and more readable.

### 6. **Complex Prompt Structure**
- Product Overview section
- Key Features (conditional list)
- Customer Feedback Analysis
- Competitive Landscape
- Market Performance with computed values
- Structured JSON output request

## Dataset

**5 products** across different categories:
- Electronics (Headphones)
- Smart Home (Hub)
- Home Appliances (Robot Vacuum)
- Wearables (Smartwatch)
- Gaming (Mechanical Keyboard)

Each product includes:
- **5 features** (with conditional display)
- **2-3 customer reviews** (rating + text + verification status)
- **2 competitors** (name + price)
- **Market metrics** (sales trend, rating, review count, launch date)
- **Special notes** (awards, status updates, etc.)

## Quick Start

### 1. Ensure you have a `.env` file

```bash
# Already configured if you ran the simple example
cat .env
```

### 2. Run the analysis

```bash
# From project root
.venv/bin/elspeth --settings example/complex/settings.yaml
```

### 3. Check results

```bash
cat example/complex/output/analysis.csv
```

## Template Features Explained

### Conditional Sections

The prompt includes different sections based on what data is available:

```jinja2
{%- if review_1_text %}
## Customer Feedback Analysis
...
{%- endif %}
```

If there are no reviews, this entire section is omitted.

### Iterative Display

Features are displayed conditionally:

```jinja2
{%- if feature_1 %}
1. {{ feature_1 }}
{%- endif %}
{%- if feature_2 %}
2. {{ feature_2 }}
{%- endif %}
```

Only features that exist are shown, maintaining clean formatting.

### Inline Conditionals

Reviews show verification status conditionally:

```jinja2
{% if review_1_verified == "true" %}✓ Verified Purchase{% endif %}
```

### Computed Values

Market position is computed from rating:

```jinja2
{% if rating >= 4.5 %}Market Leader
{% elif rating >= 4.0 %}Strong Performer
{% elif rating >= 3.5 %}Average Performer
{% else %}Needs Improvement
{% endif %}
```

## Output Structure

The LLM is prompted to return structured JSON with:

```json
{
  "overall_sentiment": "positive|mixed|negative",
  "sentiment_score": 0-100,
  "key_strengths": [...],
  "key_weaknesses": [...],
  "feature_analysis": {
    "most_praised": [...],
    "most_criticized": [...],
    "missing_features": [...]
  },
  "competitive_position": {
    "price_competitiveness": "...",
    "value_proposition": "...",
    "differentiation": "..."
  },
  "market_insights": {
    "target_audience": "...",
    "growth_potential": "...",
    "risk_factors": [...]
  },
  "recommendations": {
    "product_team": [...],
    "marketing_team": [...],
    "pricing_strategy": "..."
  },
  "executive_summary": "..."
}
```

## Customization Ideas

### 1. Add More Data Dimensions

Extend the CSV with:
- Technical specifications
- Warranty information
- Return/refund data
- Geographic performance
- Seasonal trends

### 2. Enhance Templates with Loops

For truly dynamic lists, you can use Jinja2 loops:

```jinja2
{% for i in range(1, 11) %}
  {%- if  ("feature_" ~ i) in context %}
  - {{ context["feature_" ~ i] }}
  {%- endif %}
{% endfor %}
```

### 3. Add Filters

Create custom filters in the PromptEngine:

```python
env.filters["currency"] = lambda x: f"${x:,.2f}"
env.filters["sentiment_emoji"] = lambda x: "😊" if x > 4 else "😐" if x > 3 else "😞"
```

### 4. Use Jinja2 Macros

Define reusable template sections:

```jinja2
{% macro display_review(rating, text, verified) -%}
**Rating:** {{ rating }}/5 {% if verified %}✓{% endif %}
**Comment:** {{ text }}
{%- endmacro %}

{{ display_review(review_1_rating, review_1_text, review_1_verified) }}
```

### 5. Add Data Validation

Use template conditionals to validate data quality:

```jinja2
{%- if rating and rating > 0 %}
**Rating:** {{ rating }}/5.0
{%- else %}
**Rating:** Not yet rated
{%- endif %}
```

## Comparison with Simple Example

| Feature | Simple Example | Complex Example |
|---------|---------------|-----------------|
| Data Structure | Single-level columns | Multidimensional (features, reviews, competitors) |
| Template Logic | Basic field insertion | Conditionals, filters, computed values |
| Prompt Sections | 2 (system + user) | 7 (overview, features, reviews, competitors, etc.) |
| Field Aliases | None | 7 aliases for cleaner templates |
| Conditional Rendering | None | Extensive (features, reviews, sections) |
| Data per Row | 3 fields | 25+ fields |
| Output Structure | Free-form | Structured JSON |

## Advanced Jinja2 Features

### Whitespace Control

```jinja2
{%- if condition -%}    # Strip whitespace before and after
{% if condition %}       # Keep whitespace
{%- if condition %}      # Strip before only
```

### Default Values

```jinja2
{{ notes | default("No special notes") }}
{{ competitor_2_name | default("N/A") }}
```

### String Methods

```jinja2
{{ name | title }}       # Capitalize words
{{ name | upper }}       # ALL CAPS
{{ name | lower }}       # lowercase
{{ text | truncate(50) }} # Limit length
```

### Comparison Operators

```jinja2
{% if rating >= 4.5 %}   # Greater than or equal
{% if tier == "premium" %} # Equality
{% if total_reviews > 1000 %} # Greater than
{% if notes %} # Truthiness check
```

## Performance Considerations

- **Rate Limiting:** Set to 6 requests/minute for this example
- **Token Usage:** ~1500 max tokens per request (larger prompts)
- **Processing Time:** ~5-10 seconds per product
- **Cost:** ~$0.02-0.05 per analysis (depending on model)

## Next Steps

1. **Add Transform Plugins:** Parse JSON responses, extract specific fields
2. **Add Aggregation:** Compute category-level insights across products
3. **Add Comparison Plugins:** Compare against baseline analyses
4. **Add Custom Filters:** Create domain-specific template filters
5. **Add Validation:** Verify JSON structure in responses

## Troubleshooting

### Template Syntax Errors

If you see `UndefinedError`, check that:
1. The field name matches the CSV column (or uses correct alias)
2. The field is listed in `prompt_fields`
3. Jinja2 syntax is correct (`{{` not `{`)

### Data Not Showing

If sections are missing:
1. Check conditional logic (`{% if field %}`)
2. Verify CSV data has values in those columns
3. Check for whitespace in CSV cells

### Output Format Issues

If JSON parsing fails:
1. Check the example JSON structure in prompt
2. Verify model supports JSON output
3. Add instructions to use ```json code blocks

## Documentation

- [Jinja2 Template Designer Documentation](https://jinja.palletsprojects.com/templates/)
- [elspeth Environment Variables Guide](../../docs/ENVIRONMENT_VARIABLES.md)
- [OpenRouter Models and Pricing](https://openrouter.ai/models)
