#!/bin/bash

echo "=== Textbook Extractor Verification ==="
echo

echo "✅ Checking directory structure..."
for dir in ipchat/extract/textbook ipchat/adapters/io ipchat/schemas tests/unit tests/integration archive/schemas; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir exists"
    else
        echo "  ✗ $dir missing"
    fi
done
echo

echo "✅ Checking Python files..."
for file in ipchat/cli.py ipchat/extract/textbook/pipeline.py ipchat/extract/textbook/prompts.py ipchat/schemas/textbook.py ipchat/adapters/io/pdf.py ipchat/adapters/io/adobe_extract.py; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
    else
        echo "  ✗ $file missing"
    fi
done
echo

echo "✅ Checking schema files..."
for file in ipchat/schemas/article_evidence.schema.json ipchat/schemas/textbook_chapter.schema.json; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
    else
        echo "  ✗ $file missing"
    fi
done
echo

echo "✅ Checking test files..."
for file in tests/unit/test_textbook_schema.py tests/integration/test_textbook_cli.py; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
    else
        echo "  ✗ $file missing"
    fi
done
echo

echo "✅ Checking README updates..."
if grep -q "IPchat.git" README.md; then
    echo "  ✓ README uses correct repo name (IPchat)"
else
    echo "  ✗ README still references old repo name"
fi
echo

echo "✅ Checking for removed files..."
if [ ! -f "Dockerfile" ] && [ ! -f "docker-compose.yml" ]; then
    echo "  ✓ Root Docker files removed"
else
    echo "  ✗ Root Docker files still present"
fi
echo

echo "✅ Checking CLI command..."
if grep -q "extract-textbook" ipchat/cli.py; then
    echo "  ✓ extract-textbook command present in CLI"
else
    echo "  ✗ extract-textbook command missing from CLI"
fi
echo

echo "✅ Checking research article detection..."
if grep -q "looks_like_research_article" ipchat/extract/textbook/pipeline.py; then
    echo "  ✓ Research article detection implemented"
else
    echo "  ✗ Research article detection missing"
fi
echo

echo "=== Verification Complete ==="
echo
echo "To test the textbook extractor:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Set OPENAI_API_KEY environment variable"
echo "3. Run: python -m ipchat.cli extract-textbook --pdf <pdf> --adobe-json <json> --out outputs/"