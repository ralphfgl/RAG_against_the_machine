*This activity has been created as part of the 42 curriculum by rfeghali*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about the vLLM
codebase. It indexes the repository, retrieves the most relevant snippets for a
question with BM25, and generates a grounded answer with `Qwen/Qwen3-0.6B`. Retrieval
quality is measured with recall@k.

## Instructions

```bash
# Install dependencies
make install

# 1. Index the corpus (data/raw/ -> data/processed/)
uv run python -m src index --max_chunk_size 2000

# 2. Search a single query
uv run python -m src search "How to configure the OpenAI server?" --k 5

# 3. Search a whole dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

# 4. Generate an answer for a single query
uv run python -m src answer "How to configure the OpenAI server?" --k 5

# 5. Generate answers for a whole dataset
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions

# 6. Evaluate recall@k against a ground-truth dataset (own iteration only)
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

Other `make` targets: `run`, `debug`, `clean`, `lint`, `lint-strict`.

## System architecture

`walker` reads files from `data/raw/` → `chunker` splits them → `persist` builds and
saves a BM25 index under `data/processed/`. At query time, `retriever/search.py` loads
that index and returns the top-k `MinimalSource` locations. `generator` re-fetches the
text for those sources, builds a prompt, and calls `Qwen/Qwen3-0.6B` to produce a
grounded answer. `evaluate/recall.py` compares retrieved sources to a ground-truth
dataset to compute recall@k. Everything is orchestrated through a Fire-based CLI in
`src/__main__.py` and validated with the pydantic models in `src/models.py`.

## Chunking strategy

Two chunkers are used, both capped at `--max_chunk_size` characters (default 2000):

- **Python files**: `CodeChunker` (AST/syntax-aware splitting on function and class
  boundaries).
- **Markdown**: `RecursiveChunker` (splits on paragraph/section structure).

Both fall back to naive fixed-size slicing if the structured chunker fails on a file,
so indexing never crashes on malformed input.

## Retrieval method

Lexical retrieval with **BM25** (`bm25s`), tokenized with English stopword removal and
no stemming (stemming distorts code identifiers). Each chunk is indexed once; a query
is tokenized the same way and scored against the index to return the top-k
`(file_path, first_character_index, last_character_index)` locations.

## Performance analysis

Recall@k is reported by `evaluate` / the moulinette at k = 1, 3, 5, 10. Smaller chunk
sizes generally improve recall (finer-grained, more precise spans) at the cost of more
chunks to search; the required 80% recall@5 (docs) / 50% recall@5 (code) targets guided
the choice of `max_chunk_size` and the decision to strip stopwords without stemming.

## Design decisions

- BM25 over TF-IDF: better handling of term frequency saturation and document length
  normalization on a codebase with very uneven file sizes.
- Chunk metadata is persisted as JSONL (`data/processed/chunks.jsonl`) so the BM25
  index (pickled) and the character-offset bookkeeping stay in sync via positional
  indices.
- Retrieved context is trimmed to a fixed character budget before being handed to
  Qwen3-0.6B to stay within its usable context window.

## Challenges faced

- Recovering exact character offsets after chunking, since chunkers normalize
  whitespace: solved by re-locating each chunk's text in the original file content.
- Keeping code and prose chunking separate while sharing one pipeline and one index.
- Balancing chunk size against the moulinette's 2000-character hard limit while
  preserving enough context for the model to answer well.

## Resources

- [BM25 (bm25s) documentation](https://github.com/xhluca/bm25s)
- [Chonkie chunking library](https://github.com/chonkie-inc/chonkie)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- Course material: *RAG against the machine*, Association 42.

**AI usage**: AI assistance was used to design the chunking/retrieval pipeline
structure and review the CLI and Makefile for compliance with the subject.
All generated code was read, tested, and understood before submission.
