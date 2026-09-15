# Makefile for RAG against the machine.
# IMPORTANT: recipe lines must start with a TAB, not spaces.

NAME        := rag_against_the_machine

PYTHON      := uv run python
MAIN        := -m src

# --- Inputs ---------------------------------------------------------------
# The archive is NOT part of the repo. It is provided during review, or
# the files are already in place. All setup steps are idempotent.
ARCHIVE           := data+moulinette.tar.xz
EXTRACT_DIR       := .extracted
VLLM_ZIP          := vllm-0.10.1.zip
DATASETS_ZIP      := datasets_public.zip
MOULINETTE_ZIP    := moulinette.zip

# --- Outputs --------------------------------------------------------------
CORPUS_ROOT       := data/raw/vllm-0.10.1
UNANSWERED_DIR    := data/datasets/UnansweredQuestions
ANSWERED_DIR      := data/datasets/AnsweredQuestions
SEARCH_OUT_DIR    := data/output/search_results/UnansweredQuestions
ANSWER_OUT_DIR    := data/output/search_results_and_answer/UnansweredQuestions

DOCS_DATASET      := dataset_docs_public.json
CODE_DATASET      := dataset_code_public.json

# --- Required rules (subject) ---------------------------------------------

all: install

install: _setup_dirs _setup_corpus _setup_datasets _setup_moulinette _sync_deps
	@echo ""
	@echo "== Installation complete =="
	@echo "  Corpus:     $(CORPUS_ROOT)"
	@echo "  Datasets:   $(UNANSWERED_DIR), $(ANSWERED_DIR)"
	@echo "  Moulinette: ./moulinette"
	@echo ""
	@echo "Next: make index && make bench"

run:
	$(PYTHON) $(MAIN) --help

debug:
	$(PYTHON) -m pdb -m src --help

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf data/processed/*
	@rm -rf data/output/search_results/UnansweredQuestions/*
	@rm -rf data/output/search_results_and_answer/UnansweredQuestions/*

fclean: clean
	@rm -rf data
	@rm -rf .venv
	@rm -rf $(EXTRACT_DIR)

lint:
	uv run flake8 .
	uv run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

# --- Setup sub-steps ------------------------------------------------------

_setup_dirs:
	@mkdir -p data/raw
	@mkdir -p data/processed
	@mkdir -p $(UNANSWERED_DIR)
	@mkdir -p $(ANSWERED_DIR)
	@mkdir -p $(SEARCH_OUT_DIR)
	@mkdir -p $(ANSWER_OUT_DIR)

# Corpus: only extract if data/raw/vllm-0.10.1 is missing.
_setup_corpus:
	@if [ -d "$(CORPUS_ROOT)" ]; then \
		echo "[install] Corpus already present at $(CORPUS_ROOT)"; \
	elif [ -f "$(ARCHIVE)" ]; then \
		echo "[install] Extracting $(ARCHIVE)..."; \
		mkdir -p $(EXTRACT_DIR); \
		tar -xf $(ARCHIVE) -C $(EXTRACT_DIR); \
		unzip -q -o $(EXTRACT_DIR)/$(VLLM_ZIP) -d $(EXTRACT_DIR); \
		mv $(EXTRACT_DIR)/vllm-0.10.1 data/raw/; \
	else \
		echo "[install] ERROR: corpus not found and $(ARCHIVE) missing."; \
		echo "          Please place either data/raw/vllm-0.10.1/ or $(ARCHIVE)."; \
		exit 1; \
	fi

# Datasets: only extract if the public docs dataset is missing.
_setup_datasets:
	@if [ -f "$(UNANSWERED_DIR)/$(DOCS_DATASET)" ]; then \
		echo "[install] Datasets already present."; \
	elif [ -f "$(ARCHIVE)" ]; then \
		echo "[install] Extracting datasets..."; \
		mkdir -p $(EXTRACT_DIR); \
		[ -f "$(EXTRACT_DIR)/$(DATASETS_ZIP)" ] || tar -xf $(ARCHIVE) -C $(EXTRACT_DIR); \
		unzip -q -o $(EXTRACT_DIR)/$(DATASETS_ZIP) -d $(EXTRACT_DIR)/ds; \
		mv $(EXTRACT_DIR)/ds/public/UnansweredQuestions/* $(UNANSWERED_DIR)/; \
		mv $(EXTRACT_DIR)/ds/public/AnsweredQuestions/*   $(ANSWERED_DIR)/; \
		rm -rf $(EXTRACT_DIR)/ds; \
	else \
		echo "[install] ERROR: datasets missing and $(ARCHIVE) missing."; \
		exit 1; \
	fi

# Moulinette: prefer Fedora on modern glibc (Arch), fallback Ubuntu.
# If neither the archive nor a local binary exists, warn but do not fail:
# the evaluator will provide the moulinette themselves.
_setup_moulinette:
	@if [ -x "./moulinette" ]; then \
		echo "[install] Moulinette already present at ./moulinette"; \
	elif [ -f "$(ARCHIVE)" ]; then \
		echo "[install] Extracting moulinette..."; \
		mkdir -p $(EXTRACT_DIR); \
		[ -f "$(EXTRACT_DIR)/$(MOULINETTE_ZIP)" ] || tar -xf $(ARCHIVE) -C $(EXTRACT_DIR); \
		unzip -q -o $(EXTRACT_DIR)/$(MOULINETTE_ZIP) -d $(EXTRACT_DIR)/ml; \
		if [ -f "$(EXTRACT_DIR)/ml/moulinette_pkg/moulinette-fedora" ]; then \
			cp $(EXTRACT_DIR)/ml/moulinette_pkg/moulinette-fedora ./moulinette; \
		else \
			cp $(EXTRACT_DIR)/ml/moulinette_pkg/moulinette-ubuntu ./moulinette; \
		fi; \
		chmod +x ./moulinette; \
		rm -rf $(EXTRACT_DIR)/ml; \
	else \
		echo "[install] WARN: no moulinette available. Local evaluation disabled."; \
	fi

_sync_deps:
	@uv sync

# --- Convenience targets --------------------------------------------------

index:
	$(PYTHON) $(MAIN) index --max_chunk_size 2000

search:
	$(PYTHON) $(MAIN) search "How to configure the OpenAI server?" --k 10

search_docs:
	$(PYTHON) $(MAIN) search_dataset \
		--dataset_path $(UNANSWERED_DIR)/$(DOCS_DATASET) \
		--k 10 --save_directory $(SEARCH_OUT_DIR)

search_code:
	$(PYTHON) $(MAIN) search_dataset \
		--dataset_path $(UNANSWERED_DIR)/$(CODE_DATASET) \
		--k 10 --save_directory $(SEARCH_OUT_DIR)

search_dataset: search_docs search_code

answer:
	$(PYTHON) $(MAIN) answer "How to configure the OpenAI server?" --k 10

answer_docs:
	$(PYTHON) $(MAIN) answer_dataset \
		--student_search_results_path $(SEARCH_OUT_DIR)/$(DOCS_DATASET) \
		--save_directory $(ANSWER_OUT_DIR)

answer_code:
	$(PYTHON) $(MAIN) answer_dataset \
		--student_search_results_path $(SEARCH_OUT_DIR)/$(CODE_DATASET) \
		--save_directory $(ANSWER_OUT_DIR)

answer_dataset: answer_docs answer_code

evaluate_docs:
	@if [ -x "./moulinette" ]; then \
		./moulinette evaluate_student_search_results \
			$(SEARCH_OUT_DIR)/$(DOCS_DATASET) \
			$(ANSWERED_DIR)/$(DOCS_DATASET) \
			--k 10 --max_context_length 2000; \
	else \
		echo "No moulinette binary; cannot evaluate."; \
	fi

evaluate_code:
	@if [ -x "./moulinette" ]; then \
		./moulinette evaluate_student_search_results \
			$(SEARCH_OUT_DIR)/$(CODE_DATASET) \
			$(ANSWERED_DIR)/$(CODE_DATASET) \
			--k 10 --max_context_length 2000; \
	else \
		echo "No moulinette binary; cannot evaluate."; \
	fi

evaluate: evaluate_docs evaluate_code

# Full pipeline.
bench: index search_dataset evaluate

.PHONY: all install run debug clean fclean lint lint-strict \
        _setup_dirs _setup_corpus _setup_datasets _setup_moulinette _sync_deps \
        index search search_docs search_code search_dataset \
        answer answer_docs answer_code answer_dataset \
        evaluate evaluate_docs evaluate_code bench
