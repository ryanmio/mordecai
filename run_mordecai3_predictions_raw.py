#!/usr/bin/env python3
"""Baseline runner: *unmodified* Mordecai-3 predictions.

Reads the 45-row gold subset (has_ground_truth==1) and writes
`mordecai3_predictions_raw.csv` containing only Mordecai's own top-ranked
coordinate (lat/lon to 6 decimals).  If Mordecai returns no entities, the
row is left blank.

No abbreviation expansion, no bounding-box post-filter, and no centroid
fallbacks – this is the clean library output you can cite as the benchmark.
"""

import csv
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from elasticsearch import Elasticsearch

try:
    from mordecai3 import Geoparser  # type: ignore
except ImportError:
    sys.exit("Error: mordecai3 is not installed. Activate .venv_m3 and `pip install mordecai3`.")


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "validation - TEST-FULL-H1-final.csv"
OUTFILE = ROOT / "mordecai3_predictions_raw.csv"


def main() -> None:
    if not INPUT.exists():
        sys.exit(f"Ground-truth CSV not found: {INPUT}")

    print("Loading Mordecai 3 …")
    geo = Geoparser()

    # Elastic connection just to keep Mordecai happy (Geoparser uses DSL)
    _ = Elasticsearch(hosts=["http://localhost:9200"], request_timeout=30)

    with OUTFILE.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["subject_id", "mordecai_lat", "mordecai_lon"])
        writer.writeheader()

        total = hits = 0
        with INPUT.open(newline="") as fin:
            for row in csv.DictReader(fin):
                if row.get("has_ground_truth") != "1":
                    continue
                total += 1
                sid = row["subject_id"]
                text = row["raw_entry"]

                lat = lon = ""
                try:
                    res = geo.geoparse_doc(text, known_country="USA")
                    ents: List[Dict[str, Any]] = res.get("geolocated_ents", []) if isinstance(res, dict) else res  # type: ignore
                    if ents:
                        ent0 = ents[0]
                        lat = f"{float(ent0.get('lat', ent0.get('latitude'))):.6f}"
                        lon = f"{float(ent0.get('lon', ent0.get('longitude'))):.6f}"
                        hits += 1
                except Exception as e:
                    print(f" › Mordecai failed for {sid}: {e}")

                writer.writerow({"subject_id": sid, "mordecai_lat": lat, "mordecai_lon": lon})

    print(f"\nProcessed {total} ground-truth rows.")
    print(f"Rows with a Mordecai coordinate: {hits} / {total}")
    print("Output →", OUTFILE)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1) 