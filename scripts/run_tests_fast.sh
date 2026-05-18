#!/bin/bash
# Run fast tests only (skip slow performance tests)

cd "$(dirname "$0")/.."

source venv/bin/activate

pytest -v -m "not slow" --cov=src/api --cov-report=term-missing