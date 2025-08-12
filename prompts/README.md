# Prompts Directory

This directory is reserved for storing prompt templates and configurations. Currently, prompts are embedded within the extraction pipeline modules for better cohesion.

## Current Prompt Locations

- **Textbook Extraction**: `ipchat/extract/textbook/prompts.py`
- **Article Extraction**: `ipchat/extract/article/prompts.py` (when implemented)

## Usage

Prompts are designed to:
1. Validate document type (textbook vs research article)
2. Extract structured content according to schema
3. Preserve provenance information (page numbers, references)