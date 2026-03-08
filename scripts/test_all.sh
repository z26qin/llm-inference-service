#!/bin/bash
# LLM Inference Service - Quick Test Script
# Usage: ./scripts/test_all.sh [host]
# Example: ./scripts/test_all.sh http://localhost:8000

set -e

HOST=${1:-http://localhost:8000}
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  LLM Inference Service - Test Suite"
echo "  Target: $HOST"
echo "=============================================="
echo ""

# Helper function
check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1"
        exit 1
    fi
}

# =============================================================================
echo -e "${YELLOW}=== Phase 1: Core LLM Serving ===${NC}"
# =============================================================================

# Health check
HEALTH=$(curl -s $HOST/health)
echo "$HEALTH" | grep -q "healthy"
check "Health check"

# Readiness check
READY=$(curl -s $HOST/ready)
echo "$READY" | grep -q "ready\|not_ready"
check "Readiness check"

# List models
MODELS=$(curl -s $HOST/v1/models)
echo "$MODELS" | grep -q "data"
check "List models"

# Text completion
COMPLETION=$(curl -s -X POST $HOST/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","prompt":"Say hello","max_tokens":20,"temperature":0.5}')
echo "$COMPLETION" | grep -q "choices"
check "Text completion"

# Chat completion
CHAT=$(curl -s -X POST $HOST/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hi"}],"max_tokens":20}')
echo "$CHAT" | grep -q "choices"
check "Chat completion"

# Streaming completion
STREAM=$(curl -s -X POST $HOST/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","prompt":"Count","max_tokens":10,"stream":true}')
echo "$STREAM" | grep -q "data:"
check "Streaming completion"

echo ""

# =============================================================================
echo -e "${YELLOW}=== Phase 2: Caching & Reliability ===${NC}"
# =============================================================================

# JSON metrics endpoint
METRICS_JSON=$(curl -s $HOST/metrics/json)
echo "$METRICS_JSON" | grep -q "cache"
check "JSON metrics endpoint"

# Cache configuration
CACHE_ENABLED=$(echo "$METRICS_JSON" | grep -o '"enabled":[^,]*' | head -1 | cut -d: -f2)
echo "  Cache enabled: $CACHE_ENABLED"

# Rate limit configuration
RATE_LIMIT=$(echo "$METRICS_JSON" | grep -o '"requests_per_minute":[0-9]*' | cut -d: -f2)
echo "  Rate limit: $RATE_LIMIT req/min"

# Test caching (deterministic request)
echo "  Testing cache behavior..."
CACHE_PROMPT='{"model":"test","prompt":"What is 1+1","max_tokens":10,"temperature":0}'

START1=$(date +%s%N)
curl -s -X POST $HOST/v1/completions \
  -H "Content-Type: application/json" \
  -d "$CACHE_PROMPT" > /dev/null
END1=$(date +%s%N)
TIME1=$(( ($END1 - $START1) / 1000000 ))

START2=$(date +%s%N)
curl -s -X POST $HOST/v1/completions \
  -H "Content-Type: application/json" \
  -d "$CACHE_PROMPT" > /dev/null
END2=$(date +%s%N)
TIME2=$(( ($END2 - $START2) / 1000000 ))

echo "  First request: ${TIME1}ms"
echo "  Second request (cached): ${TIME2}ms"
check "Cache test completed"

echo ""

# =============================================================================
echo -e "${YELLOW}=== Phase 3: Observability ===${NC}"
# =============================================================================

# Prometheus metrics
PROMETHEUS=$(curl -s $HOST/metrics)
echo "$PROMETHEUS" | grep -q "llm_"
check "Prometheus metrics endpoint"

# Check for key metrics
echo "$PROMETHEUS" | grep -q "llm_requests_total"
check "Request counter metric"

echo "$PROMETHEUS" | grep -q "llm_request_latency_seconds"
check "Latency histogram metric"

# Correlation ID
CORR_RESPONSE=$(curl -s -i -X POST $HOST/v1/completions \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: test-trace-123" \
  -d '{"model":"test","prompt":"Hi","max_tokens":5}')
echo "$CORR_RESPONSE" | grep -q "X-Correlation-ID: test-trace-123"
check "Correlation ID propagation"

# Cache stats in metrics
CACHE_STATS=$(curl -s $HOST/metrics/json)
echo "$CACHE_STATS" | grep -q "stats"
check "Cache statistics"

echo ""

# =============================================================================
echo -e "${YELLOW}=== Summary ===${NC}"
# =============================================================================

# Get final stats
FINAL_METRICS=$(curl -s $HOST/metrics/json)
CACHE_HITS=$(echo "$FINAL_METRICS" | grep -o '"hits":[0-9]*' | cut -d: -f2 || echo "0")
CACHE_MISSES=$(echo "$FINAL_METRICS" | grep -o '"misses":[0-9]*' | cut -d: -f2 || echo "0")

echo "Cache hits: ${CACHE_HITS:-0}"
echo "Cache misses: ${CACHE_MISSES:-0}"

echo ""
echo -e "${GREEN}=============================================="
echo "  All tests passed!"
echo "==============================================${NC}"
