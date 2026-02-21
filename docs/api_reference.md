# API Reference

## Base URL

```
http://localhost:8000/api/v1
```

## Interactive Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Authentication

No authentication required for MVP. Add authentication for production deployments.

## Endpoints Overview

| Category | Endpoints |
|----------|-----------|
| Transformers | CRUD operations for transformers |
| DGA | DGA analysis and records |
| Health | Health index calculation |
| Predictions | RUL and failure probability |
| Alerts | Alert management |
| Reports | PDF report generation |

---

## Transformers

### List all transformers

```http
GET /transformers
```

**Query Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| skip | int | Number of records to skip | 0 |
| limit | Max records to return | 100 |
| search | Optional search term for name or serial number | None |

**Example Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/transformers?skip=0&limit=10"
```

**Example Response:**

```json
[
  {
    "id": 1,
    "name": "TR-MAIN-001",
    "serial_number": "SN-2015-001",
    "manufacturer": "ABB",
    "rated_mva": 25,
    "rated_voltage_kv": 138,
    "cooling_type": "ONAF",
    "location": "Downtown Substation",
    "manufacture_date": "2015-03-15",
    "installation_date": "2015-06-01",
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00"
  }
]
```

### Get transformer by ID

```http
GET /transformers/{transformer_id}
```

**Example Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/transformers/1"
```

### Create transformer

```http
POST /transformers
```

**Request Body:**

```json
{
  "name": "TR-MAIN-001",
  "serial_number": "SN-2015-001",
  "manufacturer": "ABB",
  "rated_mva": 25,
  "rated_voltage_kv": 138,
  "cooling_type": "ONAF",
  "location": "Downtown Substation",
  "manufacture_date": "2015-03-15",
  "installation_date": "2015-06-01"
}
```

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/v1/transformers" \
  -H "Content-Type: application/json" \
  -d '{"name":"TR-MAIN-001","serial_number":"SN-2015-001","manufacturer":"ABB","rated_mva":25,"rated_voltage_kv":138,"cooling_type":"ONAF","location":"Downtown Substation"}'
```

### Update transformer

```http
PUT /transformers/{transformer_id}
```

**Request Body:**

```json
{
  "name": "TR-MAIN-001-UPDATED",
  "location": "New Location"
}
```

### Delete transformer

```http
DELETE /transformers/{transformer_id}
```

---

## DGA Analysis

### Analyze DGA sample

```http
POST /dga/analyze
```

**Request Body:**

```json
{
  "h2": 150,
  "ch4": 200,
  "c2h2": 10,
  "c2h4": 50,
  "c2h6": 30,
  "co": 250,
  "co2": 1500,
  "method": "multi"
}
```

**Parameters:**

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| h2 | float | Hydrogen (ppm) | Yes |
| ch4 | float | Methane (ppm) | Yes |
| c2h2 | float | Acetylene (ppm) | Yes |
| c2h4 | float | Ethylene (ppm) | Yes |
| c2h6 | float | Ethane (ppm) | Yes |
| co | float | Carbon Monoxide (ppm) | Yes |
| co2 | float | Carbon Dioxide (ppm) | Yes |
| o2 | float | Oxygen (ppm) | No |
| n2 | float | Nitrogen (ppm) | No |
| method | string | Analysis method (duval, rogers, iec, key_gas, multi) | No |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/v1/dga/analyze" \
  -H "Content-Type: application/json" \
  -d '{"h2":150,"ch4":200,"c2h2":10,"c2h4":50,"c2h6":30,"co":250,"co2":1500,"method":"multi"}'
```

**Example Response:**

```json
{
  "fault_type": "T2",
  "fault_type_name": "Thermal Fault 300-700°C",
  "fault_confidence": 0.75,
  "tdcg": 2190,
  "explanation": "Medium-temperature thermal fault (300-700°C). The dominant gases are CH4 and CO indicating thermal decomposition of oil and paper insulation.",
  "method_results": {
    "duval": {
      "fault_type": "T2",
      "confidence": 0.82
    },
    "rogers": {
      "fault_type": "T2",
      "codes": "0 1 1 1"
    },
    "iec": {
      "fault_type": "T2",
      "codes": "A 0 1 1"
    },
    "key_gas": {
      "fault_type": "T1-T2",
      "dominant_gases": ["CH4", "CO"]
    }
  },
  "gas_percentages": {
    "h2": 6.85,
    "ch4": 9.13,
    "c2h2": 0.46,
    "c2h4": 2.28,
    "c2h6": 1.37,
    "co": 11.42,
    "co2": 68.49
  }
}
```

### Get DGA history for transformer

```http
GET /dga/history/{transformer_id}
```

**Query Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| skip | int | Records to skip | 0 |
| limit | int | Max records | 100 |
| start_date | date | Filter start date | None |
| end_date | date | Filter end date | None |

**Example Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/dga/history/1?limit=10"
```

### Add DGA record

```http
POST /dga/records
```

**Request Body:**

```json
{
  "transformer_id": 1,
  "sample_date": "2024-01-15",
  "h2": 150,
  "ch4": 200,
  "c2h2": 10,
  "c2h4": 50,
  "c2h6": 30,
  "co": 250,
  "co2": 1500,
  "o2": 15000,
  "n2": 50000,
  "remarks": "Routine sample"
}
```

### Batch analyze DGA

```http
POST /dga/batch-analyze
```

**Request Body:**

```json
{
  "samples": [
    {"h2": 150, "ch4": 200, "c2h2": 10, "c2h4": 50, "c2h6": 30, "co": 250, "co2": 1500},
    {"h2": 100, "ch4": 150, "c2h2": 5, "c2h4": 30, "c2h6": 20, "co": 200, "co2": 1200}
  ]
}
```

---

## Health Index

### Get health index

```http
GET /health/{transformer_id}
```

**Example Request:**

```bash
curl -X GET "http://localhost:8000/api/v1/health/1"
```

**Example Response:**

```json
{
  "transformer_id": 1,
  "health_index": 78.5,
  "category": "GOOD",
  "category_color": "#27ae60",
  "component_scores": {
    "dga": 85.0,
    "oil_quality": 75.0,
    "electrical": 80.0,
    "age": 70.0,
    "loading": 82.0
  },
  "weights_used": {
    "dga": 0.35,
    "oil_quality": 0.20,
    "electrical": 0.15,
    "age": 0.15,
    "loading": 0.15
  },
  "risk_level": "MEDIUM",
  "confidence": 0.92,
  "calculation_date": "2024-01-15T10:30:00"
}
```

### Calculate health index

```http
POST /health/calculate
```

**Request Body:**

```json
{
  "transformer_id": 1,
  "include_components": true
}
```

### Get health trend

```http
GET /health/{transformer_id}/trend
```

**Query Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| days | int | Number of days of history | 365 |

### Get fleet health overview

```http
GET /health/fleet/overview
```

---

## Predictions

### Get RUL estimate

```http
GET /predictions/rul/{transformer_id}
```

**Example Response:**

```json
{
  "transformer_id": 1,
  "rul_months": 60,
  "rul_years": 5.0,
  "confidence": 0.85,
  "method": "degradation_curve",
  "factors": {
    "health_index": 78.5,
    "dga_trend": "stable",
    "loading_history": "normal",
    "age_years": 10
  },
  "calculated_at": "2024-01-15T10:30:00"
}
```

### Get failure probability

```http
GET /predictions/failure-probability/{transformer_id}
```

**Example Response:**

```json
{
  "transformer_id": 1,
  "probability_30_days": 0.02,
  "probability_60_days": 0.05,
  "probability_90_days": 0.08,
  "risk_level": "LOW",
  "factors": {
    "health_index": 78.5,
    "fault_type": "T2",
    "tdcg_trend": "increasing",
    "hot_spots": 2
  },
  "calculated_at": "2024-01-15T10:30:00"
}
```

### Forecast gas trends

```http
GET /predictions/gas-forecast/{transformer_id}
```

**Query Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| gas | string | Gas to forecast (h2, ch4, etc.) | all |
| days | int | Forecast horizon | 90 |

**Example Response:**

```json
{
  "transformer_id": 1,
  "forecasts": {
    "h2": {
      "current": 150,
      "predicted_30_days": 165,
      "predicted_60_days": 180,
      "predicted_90_days": 195,
      "trend": "increasing",
      "rate_ppm_per_month": 0.5
    }
  },
  "calculated_at": "2024-01-15T10:30:00"
}
```

### Detect anomalies

```http
GET /predictions/anomalies/{transformer_id}
```

---

## Alerts

### List alerts

```http
GET /alerts
```

**Query Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| transformer_id | int | Filter by transformer | All |
| priority | string | Filter by priority (CRITICAL, HIGH, MEDIUM, LOW) | All |
| acknowledged | bool | Filter by acknowledged status | None |
| page | int | Page number | 1 |
| page_size | int | Items per page | 20 |

**Example Response:**

```json
{
  "alerts": [
    {
      "id": 1,
      "transformer_id": 1,
      "priority": "HIGH",
      "category": "DGA",
      "title": "Rising Acetylene Level",
      "message": "C2H2 has increased by 25% from previous sample",
      "acknowledged": false,
      "acknowledged_by": null,
      "acknowledged_at": null,
      "created_at": "2024-01-15T10:30:00",
      "resolved_at": null
    }
  ],
  "total_count": 10,
  "page": 1,
  "page_size": 20
}
```

### Get alert by ID

```http
GET /alerts/{alert_id}
```

### Acknowledge alert

```http
POST /alerts/{alert_id}/acknowledge
```

**Request Body:**

```json
{
  "acknowledged_by": "operator@example.com"
}
```

### Resolve alert

```http
POST /alerts/{alert_id}/resolve
```

### Get alert summary

```http
GET /alerts/summary
```

**Example Response:**

```json
{
  "total": 15,
  "critical": 2,
  "high": 5,
  "medium": 6,
  "low": 2,
  "unacknowledged": 8,
  "by_category": {
    "DGA": 8,
    "THERMAL": 3,
    "HEALTH": 2,
    "LOADING": 2
  }
}
```

---

## Reports

### Generate report

```http
POST /reports/generate/{transformer_id}
```

**Query Parameters:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| format | string | Report format (PDF, HTML) | PDF |
| include_charts | bool | Include charts | true |
| include_recommendations | bool | Include recommendations | true |
| include_historical_data | bool | Include historical data | true |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/v1/reports/generate/1?format=PDF&include_charts=true"
```

**Example Response:**

```json
{
  "report_id": "550e8400-e29b-41d4-a716-446655440000",
  "transformer_id": 1,
  "transformer_name": "TR-MAIN-001",
  "status": "GENERATING",
  "message": "Report generation started",
  "created_at": "2024-01-15T10:30:00"
}
```

### Get report status

```http
GET /reports/status/{report_id}
```

### Download report

```http
GET /reports/download/{report_id}
```

Returns the generated report file.

### List reports

```http
GET /reports
```

---

## Error Responses

All endpoints may return the following error responses:

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid input parameters |
| 404 | Not Found - Resource not found |
| 500 | Internal Server Error |

**Error Response Example:**

```json
{
  "detail": "Transformer not found"
}
```
