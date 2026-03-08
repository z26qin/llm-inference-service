# LLM Inference Service - Autoscaling Strategy

This document describes the autoscaling strategy, metrics, and thresholds for the LLM Inference Service on Kubernetes.

## Overview

The service uses Kubernetes Horizontal Pod Autoscaler (HPA) to automatically scale based on:
1. **Resource metrics** (CPU, Memory) - built-in K8s metrics
2. **Custom metrics** (request rate, latency, in-flight requests) - via Prometheus Adapter

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Kubernetes Cluster                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│  │  LLM Pod 1   │     │  LLM Pod 2   │     │  LLM Pod N   │       │
│  │  :8000       │     │  :8000       │     │  :8000       │       │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘       │
│         │                    │                    │                │
│         └────────────────────┼────────────────────┘                │
│                              │                                      │
│                    ┌─────────▼─────────┐                           │
│                    │    Service (LB)   │                           │
│                    │   ClusterIP:80    │                           │
│                    └─────────┬─────────┘                           │
│                              │                                      │
│  ┌───────────────────────────┼───────────────────────────────────┐ │
│  │                           │        Monitoring Stack           │ │
│  │  ┌────────────────┐  ┌────▼─────┐  ┌────────────────────┐    │ │
│  │  │    HPA         │◄─│Prometheus│◄─│  Prometheus        │    │ │
│  │  │ (autoscaler)   │  │ Adapter  │  │  (metrics store)   │    │ │
│  │  └───────┬────────┘  └──────────┘  └────────────────────┘    │ │
│  │          │                                                    │ │
│  │          ▼                                                    │ │
│  │  ┌────────────────┐                                          │ │
│  │  │  Deployment    │                                          │ │
│  │  │ (scale target) │                                          │ │
│  │  └────────────────┘                                          │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Scaling Metrics

### 1. CPU Utilization (Primary)

| Setting | Value | Rationale |
|---------|-------|-----------|
| Target | 70% | Leave headroom for burst traffic |
| Scale Up | >70% for 60s | Sustained high CPU triggers scale up |
| Scale Down | <50% for 5m | Aggressive scale down prevention |

**Why 70%?**
- LLM inference is CPU/GPU intensive
- Below 70% ensures new requests don't queue
- Above 70% risks latency spikes

### 2. Memory Utilization (Safety)

| Setting | Value | Rationale |
|---------|-------|-----------|
| Target | 80% | Prevent OOM kills |
| Scale Up | >80% for 60s | Memory pressure triggers scale |

**Why 80%?**
- Model weights consume significant memory
- KV cache grows with concurrent requests
- Buffer prevents OOM during spikes

### 3. In-Flight Requests (Custom)

| Setting | Value | Rationale |
|---------|-------|-----------|
| Target | 10 per pod | Balance concurrency vs latency |
| Scale Up | >10 avg for 60s | Queue building up |

**Why 10 requests?**
- Optimal batch size for vLLM continuous batching
- Too high = latency degradation
- Too low = underutilization

### 4. Request Rate (Custom)

| Setting | Value | Rationale |
|---------|-------|-----------|
| Target | 5 req/s per pod | Throughput-based scaling |
| Scale Up | >5 req/s avg | Proactive scaling |

**Why 5 req/s?**
- Based on average generation time (~200ms)
- Accounts for prompt processing overhead
- Adjust based on your model's performance

### 5. P95 Latency (Custom)

| Setting | Value | Rationale |
|---------|-------|-----------|
| Target | 2 seconds | User experience threshold |
| Scale Up | >2s for 2m | Latency SLO violation |

**Why 2 seconds?**
- Acceptable latency for interactive use
- Streaming mitigates perceived latency
- Adjust based on your SLO

## Scaling Behavior

### Scale Up Policy

```yaml
scaleUp:
  stabilizationWindowSeconds: 60    # Wait 60s before scaling
  policies:
    - type: Pods
      value: 2                       # Add up to 2 pods
      periodSeconds: 60              # Per minute
    - type: Percent
      value: 100                     # Or double the pods
      periodSeconds: 60
  selectPolicy: Max                  # Use whichever adds more pods
```

**Rationale:**
- Fast response to traffic spikes
- 60s stabilization prevents flapping
- Doubling allows rapid scale for sudden load

### Scale Down Policy

```yaml
scaleDown:
  stabilizationWindowSeconds: 300   # Wait 5 minutes
  policies:
    - type: Pods
      value: 1                       # Remove 1 pod at a time
      periodSeconds: 120             # Every 2 minutes
  selectPolicy: Min                  # Conservative removal
```

**Rationale:**
- Slow scale down prevents thrashing
- 5-minute window smooths traffic variations
- Gradual removal maintains stability

## Replica Bounds

| Setting | Value | Rationale |
|---------|-------|-----------|
| Min Replicas | 1 | Cost optimization (dev/staging) |
| Max Replicas | 10 | Resource limits / budget cap |

**Production Recommendations:**
- **Min Replicas: 2** - High availability
- **Max Replicas:** Based on GPU/budget constraints

## Custom Metrics Setup

### Prerequisites

1. **Prometheus** - Metrics collection
2. **Prometheus Adapter** - Expose metrics to K8s API

### Installation

```bash
# Add Helm repos
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus (if not already installed)
helm install prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace

# Install Prometheus Adapter with custom rules
helm install prometheus-adapter prometheus-community/prometheus-adapter \
  -n monitoring \
  -f k8s/prometheus-adapter.yaml
```

### Verify Custom Metrics

```bash
# List available custom metrics
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1 | jq

# Query specific metric
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/llm-inference/pods/*/llm_requests_in_progress" | jq
```

## Thresholds Summary

| Metric | Scale Up Threshold | Scale Down Threshold | Window |
|--------|-------------------|---------------------|--------|
| CPU | >70% | <50% | 60s/5m |
| Memory | >80% | <60% | 60s/5m |
| In-Flight Requests | >10/pod | <5/pod | 60s/5m |
| Request Rate | >5 req/s/pod | <2 req/s/pod | 60s/5m |
| P95 Latency | >2s | <1s | 2m/5m |

## GPU Considerations

For GPU-based deployments:

1. **GPU Utilization Metric** (NVIDIA DCGM)
   ```yaml
   - type: Pods
     pods:
       metric:
         name: DCGM_FI_DEV_GPU_UTIL
       target:
         type: AverageValue
         averageValue: "80"
   ```

2. **GPU Memory Metric**
   ```yaml
   - type: Pods
     pods:
       metric:
         name: DCGM_FI_DEV_FB_USED_PERCENT
       target:
         type: AverageValue
         averageValue: "80"
   ```

3. **Install NVIDIA GPU Operator**
   ```bash
   helm install gpu-operator nvidia/gpu-operator \
     -n gpu-operator --create-namespace
   ```

## Testing Autoscaling

### 1. Generate Load

```bash
# Using Locust
locust -f tests/load/locustfile.py \
  --host http://llm.local \
  --headless -u 50 -r 10 --run-time 5m
```

### 2. Watch HPA

```bash
# Watch HPA status
kubectl get hpa llm-inference-hpa -n llm-inference -w

# Detailed HPA status
kubectl describe hpa llm-inference-hpa -n llm-inference
```

### 3. Watch Pods

```bash
kubectl get pods -n llm-inference -w
```

### 4. Check Metrics

```bash
# Via kubectl top
kubectl top pods -n llm-inference

# Via Prometheus
curl -s "http://prometheus:9090/api/v1/query?query=llm_requests_in_progress" | jq
```

## Troubleshooting

### HPA Not Scaling

1. **Check metrics availability:**
   ```bash
   kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1
   ```

2. **Check HPA conditions:**
   ```bash
   kubectl describe hpa llm-inference-hpa -n llm-inference
   ```

3. **Check Prometheus Adapter logs:**
   ```bash
   kubectl logs -n monitoring -l app=prometheus-adapter
   ```

### Scaling Too Aggressively

- Increase `stabilizationWindowSeconds`
- Reduce `value` in scale policies
- Adjust metric thresholds

### Not Scaling Fast Enough

- Decrease `stabilizationWindowSeconds`
- Increase `value` in scale policies
- Use `selectPolicy: Max` for scale up

## Cost Optimization

### Development/Staging
```yaml
minReplicas: 1
maxReplicas: 3
```

### Production
```yaml
minReplicas: 2  # HA
maxReplicas: 10 # Budget cap
```

### Scheduled Scaling (KEDA)

For predictable traffic patterns, use KEDA with cron scaler:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-inference-scheduled
spec:
  scaleTargetRef:
    name: llm-inference
  minReplicaCount: 1
  maxReplicaCount: 10
  triggers:
    # Scale up during business hours
    - type: cron
      metadata:
        timezone: America/New_York
        start: "0 8 * * 1-5"   # 8 AM Mon-Fri
        end: "0 18 * * 1-5"    # 6 PM Mon-Fri
        desiredReplicas: "5"
    # Scale down nights/weekends
    - type: cron
      metadata:
        timezone: America/New_York
        start: "0 18 * * 1-5"
        end: "0 8 * * 1-5"
        desiredReplicas: "2"
```

## Alerts

See `k8s/servicemonitor.yaml` for alert rules:

- **LLMHighErrorRate** - Error rate >5%
- **LLMHighLatency** - P95 latency >5s
- **LLMServiceDown** - Service unavailable
- **LLMCacheHitRateLow** - Cache hit rate <30%
- **LLMRateLimitingActive** - Rate limiting triggered

## References

- [Kubernetes HPA](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Prometheus Adapter](https://github.com/kubernetes-sigs/prometheus-adapter)
- [KEDA](https://keda.sh/)
- [vLLM Performance Tuning](https://docs.vllm.ai/en/latest/serving/performance.html)
