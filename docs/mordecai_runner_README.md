# Mordecai v2 Runner

This repo ships a lightweight script, `run_mordecai_predictions.py`, that generates latitude/longitude predictions for the ground-truth subset of **`validation - TEST-FULL-H1-final.csv`**.

## Installation (Python 3)

```bash
# 1. Create/activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the minimal dependencies
pip install mordecai==2.1.0 spacy geographiclib

# 3. Download the large spaCy NLP model used by Mordecai
python -m spacy download en_core_web_lg
```

> No Elasticsearch service is required for this runner.

## Usage

From the repository root:

```bash
python run_mordecai_predictions.py
```

The script will create **`mordecai_predictions.csv`** (subject_id, mordecai_lat, mordecai_lon) in the repo root and print a short success summary.

## Using Mordecai 3 (optional, higher accuracy)

A parallel script `run_mordecai3_predictions.py` targets the newer library.
The steps are the same but use a separate virtual-env so the TF/NumPy
versions don't clash with v2:

```bash
# 1. Create and activate env dedicated to v3
python3 -m venv .venv_m3
source .venv_m3/bin/activate

# 2. Install requirements
pip install mordecai3 spacy==3.7.4
python -m spacy download en_core_web_trf

# 3. Make sure the Elasticsearch-7 container with the GeoNames index is running (see above).

# 4. Run the script
python run_mordecai3_predictions.py
```

The output file `mordecai3_predictions.csv` carries the same three-column
schema as before. 