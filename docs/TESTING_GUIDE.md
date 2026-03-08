# LLM Inference Service - Testing Guide

This guide covers testing all three phases of the LLM Inference Service.

## Prerequisites

```bash
# Install dependencies
pip install -e ".[dev]"

# Start Redis (required for Phase 2 caching/rate limiting)
docker run -d --name redis -p 6379:6379 redis:latest

# Or skip Redis (service will use in-memory fallbacks)
```

## Start the Server

```bash
# Development mode (with auto-reload)
DEBUG=true python -m app.main

# Or production mode
python -m app.main

# Custom port (if 8000 is in use)
PORT=8001 python -m app.main
```

The server will start at `http://localhost:8000` (or your custom port).

---

## Phase 1: Core LLM Serving

### 1.1 Health Check

```bash
# Basic health (liveness probe)
curl http://localhost:8000/health

# Expected: {"status":"healthy"}
```

### 1.2 Readiness Check

```bash
curl http://localhost:8000/ready

# Expected: {"status":"ready","checks":{"engine":true,"redis":true/null},"model":"..."}
```

### 1.3 List Models

```bash
curl http://localhost:8000/v1/models

# Expected: {"object":"list","data":[{"id":"...","object":"model",...}]}
```

### 1.4 Text Completion

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
  }'

# Expected: {"id":"cmpl-...","choices":[{"text":"...","index":0,...}],"usage":{...}}
```

### 1.5 Chat Completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'

# Expected: {"id":"chatcmpl-...","choices":[{"message":{"role":"assistant","content":"..."},...}]}
```

### 1.6 Streaming Completion

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "test",
    "prompt": "Count from 1 to 5:",
    "max_tokens": 50,
    "stream": true
  }'

# Expected: Server-Sent Events with data chunks, ending with data: [DONE]
```

### 1.7 Streaming Chat

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "Tell me a short joke"}],
    "max_tokens": 100,
    "stream": true
  }'

# Expected: Server-Sent Events with delta content
```

---

## Phase 2: Caching & Reliability

### 2.1 Redis Connection

```bash
# Check if Redis is connected
curl http://localhost:8000/ready | jq .checks.redis

# Expected: true (if Redis is running) or null (if not configured)
```

### 2.2 Response Caching

Cache only works for deterministic requests (temperature <= 0.01, n=1, non-streaming).

```bash
# First request - cache miss (will take time to generate)
time curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "prompt": "What is the capital of France?",
    "max_tokens": 30,
    "temperature": 0
  }'

# Second request - cache hit (should be instant)
time curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "prompt": "What is the capital of France?",
    "max_tokens": 30,
    "temperature": 0
  }'

# Verify cache stats
curl http://localhost:8000/metrics/json | jq .cache
```

### 2.3 Rate Limiting

```bash
# Send many requests quickly to trigger rate limiting
for i in {1..65}; do
  curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/v1/models
done

# After ~60 requests (default limit), you should see 429 responses
# Check for rate limit headers in response:
curl -i http://localhost:8000/v1/models

# Headers to look for:
# X-RateLimit-Limit: 60
# X-RateLimit-Remaining: 59
# X-RateLimit-Reset: <timestamp>
```

### 2.4 Timeout Handling

```bash
# Set a short timeout for testing
GENERATION_TIMEOUT_SECONDS=1 python -m app.main &

# Request that would take longer than 1 second
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "prompt": "Write a very long essay about...",
    "max_tokens": 500
  }'

# Expected: 504 Gateway Timeout with error message
```

### 2.5 Cache Statistics

```bash
curl http://localhost:8000/metrics/json | jq

# Expected output:
# {
#   "engine": {"ready": true, "model": "..."},
#   "cache": {"enabled": true, "connected": true, "stats": {...}},
#   "rate_limit": {"enabled": true, "requests_per_minute": 60, ...}
# }
```

---

## Phase 3: Observability

### 3.1 Prometheus Metrics

```bash
curl http://localhost:8000/metrics

# Expected: Prometheus text format metrics like:
# # HELP llm_requests_total Total number of requests
# # TYPE llm_requests_total counter
# llm_requests_total{endpoint="/v1/completions",status="success"} 5.0
# ...
```

Key metrics to check:
- `llm_requests_total` - Request counts by endpoint and status
- `llm_request_latency_seconds` - Request latency histogram
- `llm_tokens_generated_total` - Total tokens generated
- `llm_cache_hits_total` / `llm_cache_misses_total` - Cache performance
- `llm_rate_limit_hits_total` - Rate limiting events
- `llm_errors_total` - Error counts by type

### 3.2 JSON Metrics (Debugging)

```bash
curl http://localhost:8000/metrics/json | jq

# More readable format for debugging
```

### 3.3 Correlation ID Tracing

```bash
# Send request with correlation ID
curl -i -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: my-trace-123" \
  -d '{"model":"test","prompt":"Hi","max_tokens":10}'

# Response headers will include:
# X-Correlation-ID: my-trace-123
# X-Request-ID: my-trace-123

# Check server logs - all log entries will have correlation_id field
```

### 3.4 Structured Logging

```bash
# Start with JSON logging (default in production)
LOG_FORMAT=json LOG_LEVEL=DEBUG python -m app.main

# Logs will be structured JSON:
# {"event":"HTTP request completed","http_method":"POST","http_path":"/v1/completions",...}

# Start with console logging (development)
LOG_FORMAT=console LOG_LEVEL=DEBUG python -m app.main

# Logs will be human-readable with colors
```

### 3.5 Load Testing with Locust

```bash
# Install Locust
pip install locust

# Start Locust web UI
locust -f tests/load/locustfile.py --host http://localhost:8000

# Open http://localhost:8089 in browser
# Configure:
#   - Number of users: 10
#   - Spawn rate: 2
# Click "Start Swarming"
```

**Headless mode (CI/CD):**

```bash
# Run for 60 seconds with 10 users
locust -f tests/load/locustfile.py \
  --host http://localhost:8000 \
  --headless \
  -u 10 \
  -r 2 \
  --run-time 60s

# With HTML report
locust -f tests/load/locustfile.py \
  --host http://localhost:8000 \
  --headless \
  -u 10 \
  -r 2 \
  --run-time 60s \
  --html report.html
```

**Available user classes:**

| Class | Description | Wait Time |
|-------|-------------|-----------|
| `LLMInferenceUser` | Normal usage with mixed endpoints | 1-3s |
| `HighLoadUser` | Stress testing with rapid requests | 0.1-0.5s |
| `CacheTestUser` | Cache hit/miss behavior testing | 0.5-1s |

```bash
# Use specific user class
locust -f tests/load/locustfile.py \
  --host http://localhost:8000 \
  --class-picker  # Shows UI to select user class
```

### 3.6 Grafana Dashboard

1. **Start Prometheus** (to scrape metrics):

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'llm-inference'
    static_configs:
      - targets: ['localhost:8000']
```

```bash
docker run -d --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

2. **Start Grafana**:

```bash
docker run -d --name grafana \
  -p 3000:3000 \
  grafana/grafana
```

3. **Import Dashboard**:
   - Open http://localhost:3000 (admin/admin)
   - Go to Dashboards > Import
   - Upload `grafana/dashboard.json`
   - Select Prometheus data source
   - Click Import

4. **Dashboard Panels**:
   - Service Overview: Engine/Redis status, request rates
   - Latency: p50/p95/p99 percentiles
   - Token Throughput: Tokens/second metrics
   - Cache Performance: Hit rate, hits vs misses
   - Rate Limiting & Errors: Rate limit hits, error rates

---

## Quick Test Script

Save this as `test_all.sh`:

```bash
#!/bin/bash
set -e

HOST=${1:-http://localhost:8000}

echo "=== Phase 1: Core LLM Serving ==="

echo -n "Health check: "
curl -s $HOST/health | jq -r .status

echo -n "Readiness: "
curl -s $HOST/ready | jq -r .status

echo -n "List models: "
curl -s $HOST/v1/models | jq -r '.data[0].id'

echo "Completion: "
curl -s -X POST $HOST/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","prompt":"Hello","max_tokens":20}' | jq -r '.choices[0].text'

echo "Chat completion: "
curl -s -X POST $HOST/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hi"}],"max_tokens":20}' | jq -r '.choices[0].message.content'

echo ""
echo "=== Phase 2: Caching & Reliability ==="

echo -n "Cache enabled: "
curl -s $HOST/metrics/json | jq -r '.cache.enabled'

echo -n "Redis connected: "
curl -s $HOST/metrics/json | jq -r '.cache.connected'

echo "Testing cache (2 identical requests):"
echo -n "  Request 1: "
time -p curl -s -X POST $HOST/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","prompt":"Cache test","max_tokens":10,"temperature":0}' > /dev/null 2>&1

echo -n "  Request 2 (cached): "
time -p curl -s -X POST $HOST/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","prompt":"Cache test","max_tokens":10,"temperature":0}' > /dev/null 2>&1

echo ""
echo "=== Phase 3: Observability ==="

echo -n "Prometheus metrics available: "
curl -s $HOST/metrics | head -1

echo -n "Total requests: "
curl -s $HOST/metrics | grep "llm_requests_total" | head -1 || echo "0"

echo -n "Cache hits: "
curl -s $HOST/metrics/json | jq -r '.cache.stats.hits // 0'

echo -n "Cache misses: "
curl -s $HOST/metrics/json | jq -r '.cache.stats.misses // 0'

echo ""
echo "=== All tests passed! ==="
```

Run with:
```bash
chmod +x test_all.sh
./test_all.sh http://localhost:8000
```

---

## Troubleshooting

### Server won't start
- Check if port is in use: `lsof -i :8000`
- Use different port: `PORT=8001 python -m app.main`

### vLLM/CUDA errors on Mac
- The mock engine is automatically used on non-CUDA systems
- Set `USE_MOCK_ENGINE=true` to force mock mode

### Redis connection failed
- Check Redis is running: `docker ps | grep redis`
- Service continues with in-memory fallbacks if Redis unavailable

### Rate limiting not working
- Check if enabled: `curl localhost:8000/metrics/json | jq .rate_limit`
- Default: 60 requests/minute per IP

### Metrics not appearing
- Make some requests first to generate data
- Check `/metrics` endpoint directly
- Verify Prometheus is scraping correctly
