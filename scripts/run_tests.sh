#!/bin/bash
# Run ChurnGuard AI tests

cd "$(dirname "$0")/.."

echo "======================================================================="
echo "ChurnGuard AI - Running Tests"
echo "======================================================================="

# Activate virtual environment
source venv/bin/activate

# Run tests with coverage
pytest -v \
    --cov=src/api \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    --cov-fail-under=70

TEST_EXIT_CODE=$?

echo ""
echo "======================================================================="
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Some tests failed"
fi
echo "======================================================================="
echo ""
echo "Coverage report: htmlcov/index.html"
echo ""

exit $TEST_EXIT_CODE