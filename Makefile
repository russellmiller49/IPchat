PY ?= python

.PHONY: test-gold-invariants test fmt

# Run only the gold-standard invariant tests (fast, no network)
test-gold-invariants:
	$(PY) tools/run_invariant_tests.py

# Run all pytest tests quietly
test:
	pytest -q

# Format only changed Python files (fallback to key dirs)
CHANGED_PY := $(shell git diff --name-only --diff-filter=ACMRTUXB HEAD 2>/dev/null | grep -E '\.py$$' || true)

fmt:
	@echo "Formatting Python files..."
	@changed="$(CHANGED_PY)"; \
	if [ -n "$$changed" ]; then \
		echo "Changed files:" $$changed; \
		ruff check --fix $$changed; \
		$(PY) -m black $$changed; \
	else \
		echo "No changed files relative to HEAD; formatting key dirs"; \
		ruff check --fix tools tests utils ipchat || true; \
		$(PY) -m black tools tests utils ipchat || true; \
	fi
