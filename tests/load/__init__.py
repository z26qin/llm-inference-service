"""
Load testing module for LLM Inference Service.

Uses Locust for realistic load testing scenarios.

Usage:
    locust -f tests/load/locustfile.py --host http://localhost:8000

Or run headless:
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --headless -u 10 -r 2 --run-time 60s
"""
