# Deployment Guide

This guide covers deploying TransformerGuard to production environments.

## Deployment Options

### 1. Development Deployment

For local development:

```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run dashboard (separate terminal)
cd dashboard && streamlit run app.py
```

### 2. Docker Deployment (Recommended)

A containerized deployment is the recommended approach for production.

#### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+

#### Docker Compose Setup

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/transformerguard
      - API_HOST=0.0.0.0
      - API_PORT=8000
    depends_on:
      - db
    restart: unless-stopped

  dashboard:
    build: ./dashboard
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=transformerguard
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

#### Build and Run

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 3. Kubernetes Deployment

For large-scale deployments, Kubernetes provides orchestration:

```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: transformerguard-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: transformerguard-api
  template:
    metadata:
      labels:
        app: transformerguard-api
    spec:
      containers:
      - name: api
        image: transformerguard/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: transformerguard-secrets
              key: database-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: transformerguard-api
spec:
  selector:
    app: transformerguard-api
  ports:
  - port: 80
    targetPort: 8000
---
apiVersion: v1
kind: Ingress
metadata:
  name: transformerguard-ingress
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
spec:
  rules:
  - host: transformerguard.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: transformerguard-api
            port:
              number: 80
```

---

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| DATABASE_URL | PostgreSQL connection string | sqlite:///data/transformerguard.db | No |
| API_HOST | API server host | 0.0.0.0 | No |
| API_PORT | API server port | 8000 | No |
| LOG_LEVEL | Logging level | INFO | No |
| CORS_ORIGINS | Comma-separated CORS origins | * | No |
| SECRET_KEY | Secret key for sessions | auto-generated | No |

### Production Database Setup

For PostgreSQL:

```bash
# Create database
createdb transformerguard

# Set environment variable
export DATABASE_URL="postgresql://username:password@localhost:5432/transformerguard"
```

---

## Production Checklist

### Security

- [ ] Change default database to PostgreSQL
- [ ] Enable HTTPS/TLS
- [ ] Set up authentication (OAuth2, JWT)
- [ ] Configure firewall rules
- [ ] Use strong database passwords
- [ ] Enable API rate limiting
- [ ] Configure CORS properly

### Monitoring

- [ ] Set up application monitoring (Prometheus, Grafana)
- [ ] Configure log aggregation
- [ ] Set up health check endpoints
- [ ] Configure alerts for errors
- [ ] Set up metrics collection

### Backups

- [ ] Configure automated database backups
- [ ] Test backup restoration
- [ ] Set up off-site backup storage
- [ ] Document recovery procedures

### Performance

- [ ] Configure database connection pooling
- [ ] Set up caching (Redis)
- [ ] Configure load balancing
- [ ] Optimize static file serving

---

## Configuration Files

### settings.yaml (Production)

```yaml
app:
  name: "TransformerGuard"
  version: "1.0.0"
  debug: false
  environment: "production"

database:
  type: "postgresql"
  host: "db.example.com"
  port: 5432
  name: "transformerguard"
  pool_size: 20
  max_overflow: 40

api:
  host: "0.0.0.0"
  port: 8000
  cors_origins:
    - "https://transformerguard.example.com"
  rate_limit:
    enabled: true
    requests_per_minute: 100

dashboard:
  title: "TransformerGuard Dashboard"
  theme: "light"
  refresh_interval: 300

logging:
  level: "WARNING"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "/var/log/transformerguard/app.log"
  max_bytes: 10485760
  backup_count: 5

reports:
  output_dir: "/var/reports"
  format: "pdf"
  include_charts: true

weather:
  api_url: "https://api.open-meteo.com/v1/forecast"
  enabled: true
  api_key: "${WEATHER_API_KEY}"
```

---

## SSL/TLS Configuration

### Using Let's Encrypt with Nginx

```nginx
# /etc/nginx/sites-available/transformerguard

server {
    listen 80;
    server_name transformerguard.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name transformerguard.example.com;

    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /dashboard/ {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## Backup and Recovery

### Database Backup

```bash
# PostgreSQL backup
pg_dump -U username transformerguard > backup_$(date +%Y%m%d).sql

# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -U $DB_USER $DB_NAME > /backups/transformerguard_$DATE.sql
find /backups -type f -mtime +30 -delete
```

### Restore from Backup

```bash
psql -U username transformerguard < backup_20240115.sql
```

---

## Performance Tuning

### Database Optimization

```sql
-- Create indexes for common queries
CREATE INDEX idx_dga_transformer_date ON dga_records(transformer_id, sample_date);
CREATE INDEX idx_health_transformer_date ON health_index_records(transformer_id, calculation_date);
CREATE INDEX idx_alerts_transformer ON alerts(transformer_id, created_at);

-- Analyze tables for query optimization
ANALYZE dga_records;
ANALYZE health_index_records;
ANALYZE alerts;
```

### API Performance

```python
# Add caching to reduce database load
from functools import lru_cache

@lru_cache(maxsize=128)
def get_transformer_health(transformer_id: int):
    # Cached for 5 minutes
    return calculate_health(transformer_id)
```

---

## Troubleshooting

### Common Issues

#### High Memory Usage

```bash
# Check memory usage
docker stats

# Reduce memory by limiting workers
uvicorn src.api.main:app --workers 2 --limit-concurrency 10
```

#### Database Connection Errors

```bash
# Check database connectivity
pg_isready -h db.example.com -p 5432

# Check connection pool settings
# Increase pool_size in config
```

#### Slow API Responses

- Check database query performance
- Add database indexes
- Enable response caching
- Scale horizontally with more workers

---

## Health Check Endpoints

The API provides health check endpoints:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed
```

Response:

```json
{
  "status": "healthy",
  "database": "connected",
  "api_version": "1.0.0",
  "uptime_seconds": 3600
}
```

---

## Support

For deployment issues:
1. Check logs: `docker-compose logs -f`
2. Verify configuration
3. Test database connectivity
4. Review security settings

---

## References

- [FastAPI Deployment](https://fastapi.tiangolo.com/de/)
- [Docker Best Practices](https://docs.docker.com/develop/develop-best-practices/)
- [PostgreSQL Administration](https://www.postgresql.org/docs/current/admin.html)
