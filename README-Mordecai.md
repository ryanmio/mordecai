# Mordecai Baselines for Virginia Land Grants

This directory contains baseline geocoding approaches for the Virginia land grant dataset.

## Files

### Enhanced Mordecai-3 Baseline
- `run_mordecai3_predictions_enhanced.py` - Enhanced Mordecai-3 with text preprocessing
- `mordecai3_predictions_enhanced.csv` - Output with enhanced preprocessing
- Results: 24/45 real Mordecai hits (53%), 21 fallbacks, median 60km error

### Raw Mordecai-3 Baseline  
- `run_mordecai3_predictions_raw.py` - Vanilla Mordecai-3 without preprocessing
- `mordecai3_predictions_raw.csv` - Raw Mordecai output
- Results: 10/45 hits (22%), median 625km error

### County Centroid Baseline
- `run_county_centroid_baseline.py` - Simple county extraction + centroid lookup
- `county_centroid_predictions.csv` - County centroid coordinates
- Results: 23/45 county hits, 22 state fallbacks

## Usage

### Enhanced Mordecai-3
```bash
.venv_m3/bin/python run_mordecai3_predictions_enhanced.py
# produces mordecai3_predictions_enhanced.csv
```

### Raw Mordecai-3
```bash
.venv_m3/bin/python run_mordecai3_predictions_raw.py  
# produces mordecai3_predictions_raw.csv
```

### County Centroid
```bash
python3 run_county_centroid_baseline.py
# produces county_centroid_predictions.csv
```

## Evaluation

```bash
.venv_m3/bin/python evaluate_predictions.py <predictions_file.csv>
```

## Output Format

All prediction files follow the same format:
```
subject_id,<baseline>_lat,<baseline>_lon
test_entry_01,37.134000,-76.817500
test_entry_03,37.497500,-76.999000
...
``` 