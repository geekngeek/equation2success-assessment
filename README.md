# Transcript Analyzer

LLM-powered transcript scoring API with Pydantic validation and retry logic.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your OpenAI key to `.env`:

```
OPENAI_API_KEY=sk-...
```

## Run the app

```bash
source .venv/bin/activate
python main.py
```

Runs tests for all 5 mock LLM modes, then starts the server on port 7500.

## Run unit tests

```bash
source .venv/bin/activate
python -m pytest test_main.py -v
```

## Run model evaluation

```bash
source .venv/bin/activate
python test_prompt_models.py
```

Tests `build_prompt()` against 6 OpenAI models and reports pass rates.
