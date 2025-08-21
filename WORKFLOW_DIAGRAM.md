# Bronchmonkey Workflow Diagrams

## 🔄 Complete System Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                     BRONCHMONKEY SYSTEM                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      1. INPUT STAGE                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│   📄 PDFs              🗂️ Adobe JSONs                        │
│   (Optional)           (Required)                            │
│       │                     │                                │
│       └─────────┬───────────┘                                │
│                 ▼                                             │
│         [Extract Data]                                       │
│    python tools/medical_extractor.py                         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    2. PROCESSING STAGE                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│        🗃️ Extracted JSONs                                    │
│     (oe_final_outputs/)                                      │
│              │                                                │
│              ▼                                                │
│        [Chunking]                                            │
│    Break into 450-token pieces                               │
│              │                                                │
│              ▼                                                │
│        📑 Chunks                                             │
│     (chunks.jsonl)                                           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    3. INDEXING STAGE                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│     Chunks ──┬──► [FAISS] ──► 🔮 Vector Index              │
│              │                 (Semantic Search)             │
│              │                                                │
│              ├──► [BM25] ───► 📚 Keyword Index              │
│              │                 (Term Matching)               │
│              │                                                │
│              └──► [PostgreSQL] ► 🗄️ Database                │
│                                  (SQL Queries)               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     4. SEARCH STAGE                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│    👤 User Question                                          │
│           │                                                   │
│           ▼                                                   │
│    [Hybrid Search]                                           │
│      ├─ 50% Vector (meaning)                                │
│      ├─ 30% BM25 (keywords)                                 │
│      └─ 20% SQL (numbers)                                   │
│           │                                                   │
│           ▼                                                   │
│    📊 Ranked Results                                         │
│           │                                                   │
│           ▼                                                   │
│    [GPT-4/GPT-5]                                            │
│    Generate Answer                                           │
│           │                                                   │
│           ▼                                                   │
│    💬 Answer + Citations                                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 📝 Simple User Journey

```
YOU HAVE                    YOU WANT                   YOU DO
────────                    ────────                   ──────

New Paper (PDF)     →    Add to System         →    1. Put in folders
                                                    2. Run extractor
                                                    3. Rebuild index

Medical Question    →    Get Answer            →    1. Start Bronchmonkey
                                                    2. Type question
                                                    3. Read answer

Many Papers        →     Process All           →    1. Put in folders
                                                    2. Run batch mode
                                                    3. Wait ~30 min

Need Statistics    →     Find Numbers          →    1. Ask specific Q
                                                    2. Get cited data
                                                    3. Verify sources
```

## 🚀 Quick Start Flow

```
FIRST TIME SETUP
================
     │
     ▼
[Download Code]
     │
     ▼
[Install Tools]
pip install -r requirements.txt
     │
     ▼
[Add API Key]
Create .env file
     │
     ▼
[Build Database]
./rebuild_knowledge_base.sh
     │
     ▼
   READY!

DAILY USE
=========
     │
     ▼
[Start System]
./start.sh
     │
     ▼
[Open Browser]
localhost:8501
     │
     ▼
[Ask Questions]
Type & Enter
     │
     ▼
[Get Answers]
With citations!
```

## 🔧 Adding New Papers Flow

```
Step 1: PREPARE FILES
=====================
                PDF File                 Adobe JSON
                    │                         │
                    ▼                         ▼
            data/raw_pdfs/           data/input_articles/
                    │                         │
                    └──────────┬──────────────┘
                               │
Step 2: EXTRACT               ▼
=================   python tools/medical_extractor.py
                               │
                               ▼
                    data/oe_final_outputs/
                               │
Step 3: INDEX                  ▼
==============      ./rebuild_knowledge_base.sh
                               │
                               ▼
Step 4: USE              [Searchable!]
===========                    │
                               ▼
                        Ask questions
                         Get answers
```

## 📊 Data Flow Summary

```
INPUT FILES           PROCESSING           STORAGE            OUTPUT
───────────          ────────────         ─────────          ──────

Adobe JSON ──┐
             ├──► Extractor ──► oe_final ──► Chunker ──┐
PDF ─────────┘                                          │
                                                        ▼
                                                    Indices
                                                        │
                                                        ▼
Question ──────────────────────────────────► Search API
                                                        │
                                                        ▼
                                                    Answer
```

## 🎯 Command Decision Tree

```
What do you want to do?
           │
    ┌──────┴──────┬──────────┬────────────┬──────────┐
    │             │          │            │          │
Add Paper    Search      Check Status   Fix Issue   Update
    │             │          │            │          │
    ▼             ▼          ▼            ▼          ▼
Extract      Start UI    Status.py   See Guide   Rebuild
    │             │          │            │          │
--single     start.sh    python...   USER_GUIDE    .sh
--batch                              Troubleshoot
```

## 💡 Key Concepts Simplified

```
EXTRACTION = Reading papers and understanding them
    Input:  Paper.pdf or Paper.json
    Output: Paper.oe_final.json (structured data)

CHUNKING = Breaking into bite-sized pieces
    Input:  Full documents
    Output: Small text segments (450 words each)

INDEXING = Making searchable
    FAISS:  Understands meaning ("lung" ~ "pulmonary")
    BM25:   Finds exact words ("FEV1", "p-value")
    SQL:    Queries numbers (">15%", "<0.05")

SEARCH = Finding relevant information
    Input:  Your question
    Process: Check all three indices
    Output: Best matching chunks

ANSWER = Generating response
    Input:  Question + Found chunks
    Process: GPT-4/GPT-5 synthesis
    Output: Complete answer with citations
```