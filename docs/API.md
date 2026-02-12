# API Documentation

## Base URL
```
http://localhost:5000/api
```

---

## Endpoints

### 1. Health Check

Check if the API and model are running.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": 1706540400.123
}
```

---

### 2. Analyze Text

Analyze text or article for media bias.

**Endpoint:** `POST /analyze`

**Request Body:**
```json
{
  "text": "Article text to analyze..."
}
```

OR

```json
{
  "url": "https://example.com/news-article"
}
```

**Response (Success):**
```json
{
  "success": true,
  "data": {
    "text_preview": "First 200 characters of the text...",
    "biases": {
      "political": { "detected": true, "score": 0.89 },
      "gender": { "detected": false, "score": 0.12 },
      "entity": { "detected": true, "score": 0.67 },
      "racial": { "detected": false, "score": 0.08 },
      "religious": { "detected": false, "score": 0.15 },
      "regional": { "detected": false, "score": 0.11 },
      "sensationalism": { "detected": true, "score": 0.72 }
    },
    "detected_biases": ["political", "entity", "sensationalism"],
    "summary": "Multiple biases detected: Political Bias, Entity Bias, Sensationalism."
  },
  "processing_time_ms": 234.56
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "Either 'text' or 'url' must be provided"
}
```

**Status Codes:**
- `200`: Success
- `400`: Bad request (invalid input)
- `500`: Server error

---

### 3. Batch Analysis

Analyze multiple texts in a single request.

**Endpoint:** `POST /analyze/batch`

**Request Body:**
```json
{
  "texts": [
    "First article text...",
    "Second article text...",
    "Third article text..."
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "biases": { ... },
      "detected_biases": [...],
      "summary": "..."
    },
    {
      "biases": { ... },
      "detected_biases": [...],
      "summary": "..."
    }
  ],
  "count": 3,
  "processing_time_ms": 567.89
}
```

**Limits:**
- Maximum batch size: 20 articles
- Maximum text length: 50,000 characters per article

---

### 4. Get Bias Types

Get information about all bias types.

**Endpoint:** `GET /bias-types`

**Response:**
```json
{
  "success": true,
  "data": {
    "political": {
      "name": "Political Bias",
      "description": "Favoring or opposing political parties, ideologies, leaders, or policies",
      "examples": [...],
      "indicators": [...]
    },
    "gender": { ... },
    "entity": { ... },
    "racial": { ... },
    "religious": { ... },
    "regional": { ... },
    "sensationalism": { ... }
  }
}
```

---

### 5. Get Specific Bias Type

Get detailed information about a specific bias type.

**Endpoint:** `GET /bias-types/{bias_type}`

**Parameters:**
- `bias_type`: One of `political`, `gender`, `entity`, `racial`, `religious`, `regional`, `sensationalism`

**Response:**
```json
{
  "success": true,
  "data": {
    "name": "Political Bias",
    "description": "Favoring or opposing political parties, ideologies, leaders, or policies",
    "examples": [
      "The government's visionary policies...",
      "Opposition's failed leadership..."
    ],
    "indicators": [
      "Loaded language about political entities",
      "One-sided coverage of political events",
      "Missing context for opposing viewpoints"
    ]
  }
}
```

---

## Bias Score Interpretation

| Score Range | Interpretation |
|------------|----------------|
| 0.00 - 0.30 | Low likelihood of bias |
| 0.30 - 0.50 | Moderate - may warrant attention |
| 0.50 - 0.70 | High - bias likely present |
| 0.70 - 1.00 | Very High - strong bias indicators |

**Detection Threshold:** 0.5 (50%)

Articles with scores above 0.5 for a bias type are flagged as having that bias detected.

---

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid input parameters |
| 404 | Not Found - Resource doesn't exist |
| 500 | Internal Server Error |

---

## Rate Limits

- 60 requests per minute per IP (production)
- No rate limit in development mode

---

## Example Usage

### cURL

```bash
# Analyze text
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "The government'\''s revolutionary policies have transformed the nation..."}'

# Analyze URL
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/news-article"}'
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:5000/api/analyze",
    json={"text": "Article text here..."}
)

result = response.json()
if result["success"]:
    for bias, data in result["data"]["biases"].items():
        if data["detected"]:
            print(f"{bias}: {data['score']:.2f}")
```

### JavaScript

```javascript
const response = await fetch('http://localhost:5000/api/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Article text here...' })
});

const result = await response.json();
console.log(result.data.detected_biases);
```
