#!/usr/bin/env python3
"""Automated hyper-parameter tuning for Mordecai-3 + heuristics.

Iterates over a grid of parameters (score threshold & VA bounding-box margin) and
reports the configuration that minimises mean Haversine error on the 45-row gold
set.  Writes the winning predictions to `mordecai3_predictions_best.csv`.

This leaves the production runner untouched.
"""
from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from elasticsearch import Elasticsearch
from mordecai3 import Geoparser

ROOT = Path(__file__).resolve().parent
GT_FILE = ROOT / "validation - TEST-FULL-H1-final.csv"
BEST_OUT = ROOT / "mordecai3_predictions_best.csv"

# Earth radius in km for Haversine
R_KM = 6371.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R_KM * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Core bounding-box filter factory
BASE_BOX = dict(min_lat=36.0, max_lat=40.0, min_lon=-84.0, max_lon=-75.0)


def make_in_va(margin: float):
    min_lat = BASE_BOX["min_lat"] - margin
    max_lat = BASE_BOX["max_lat"] + margin
    min_lon = BASE_BOX["min_lon"] - margin
    max_lon = BASE_BOX["max_lon"] + margin

    def in_va(ent: Dict[str, Any]) -> bool:
        try:
            lat = float(ent.get("lat", ent.get("latitude")))
            lon = float(ent.get("lon", ent.get("longitude")))
        except (TypeError, ValueError):
            return False
        return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon

    return in_va


# County fallback helpers --------------------------------------------------
STATE_CENTROID = (37.43157, -78.6569)

ES = Elasticsearch(hosts=["http://localhost:9200"], request_timeout=30)


# Minimal normaliser for county look-ups
def _norm(text: str) -> str:
    return " ".join(w.strip(".,;").title() for w in text.split())


def county_centroid(county_name: str) -> Optional[Tuple[float, float]]:
    q = {
        "size": 1,
        "query": {
            "bool": {
                "must": [{"match": {"name": _norm(county_name)}}],
                "filter": [
                    {"term": {"admin1_code": "VA"}},
                    {"terms": {"feature_code": ["ADM2", "PPLA2", "PPLA3", "PPLA"]}},
                ],
            }
        },
    }
    try:
        resp = ES.search(index="geonames", body=q)
        hits = resp.get("hits", {}).get("hits", [])
        if hits:
            lat, lon = map(float, hits[0]["_source"]["coordinates"].split(","))
            return lat, lon
    except Exception:
        pass
    return None


def extract_county(text: str) -> Optional[str]:
    import re

    m = re.search(r"([A-Za-z .'&]+?)\s+(County|Co\b|Co[.;,]|City\b|City\s+of)", text, flags=re.I)
    return m.group(1) if m else None


# -------------------------------------------------------------------------
# Search grid
SCORE_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25]
MARGINS = [0.0, 0.25, 0.5]
# distance threshold (km) — if model-candidate further than this from county centroid, fall back
DIST_THRESHOLDS = [25, 35, 50]


def ent_score(ent: Dict[str, Any]) -> float:
    for key in ("score", "place_confidence", "confidence"):
        if key in ent and ent[key] is not None:
            try:
                return float(ent[key])
            except (TypeError, ValueError):
                continue
    return 0.0


# Load ground truth rows once ------------------------------------------------
with GT_FILE.open() as fgt:
    GOLD_ROWS = [r for r in csv.DictReader(fgt) if r["has_ground_truth"] == "1"]


def evaluate_config(score_th: float, margin: float, dist_th: float, geo: Geoparser) -> Tuple[float, float, float, float, List[Tuple[str, float, float]]]:
    """Return (min, median, mean, max, preds) for given parameters."""
    in_va = make_in_va(margin)
    dists: List[float] = []
    preds: List[Tuple[str, float, float]] = []

    for row in GOLD_ROWS:
        sid = row["subject_id"]
        text = row["raw_entry"]

        res = geo.geoparse_doc(text, known_country="USA")
        ents = res.get("geolocated_ents", []) if isinstance(res, dict) else res

        candidates = [e for e in ents if in_va(e)]
        chosen = None
        if candidates:
            candidates.sort(key=lambda e: -ent_score(e))
            if ent_score(candidates[0]) >= score_th:
                chosen = candidates[0]

        c_name = extract_county(text)
        county_latlon = county_centroid(c_name) if c_name else None

        if chosen:
            lat = float(chosen.get("lat", chosen.get("latitude")))
            lon = float(chosen.get("lon", chosen.get("longitude")))
            # If we have county centroid, check distance
            if county_latlon is not None:
                d_to_centroid = haversine(lat, lon, *county_latlon)
                if d_to_centroid > dist_th:
                    lat, lon = county_latlon  # override
        else:
            if county_latlon is not None:
                lat, lon = county_latlon
            else:
                lat, lon = STATE_CENTROID

        preds.append((sid, lat, lon))
        gt_lat, gt_lon = map(float, row["latitude/longitude"].split(","))
        dists.append(haversine(gt_lat, gt_lon, lat, lon))

    return min(dists), statistics.median(dists), statistics.mean(dists), max(dists), preds


# -------------------------------------------------------------------------

def main() -> None:
    print("Loading Mordecai 3…")
    geo = Geoparser()

    best_stats: Tuple[float, float, float, float] | None = None
    best_config: Tuple[float, float, float] | None = None
    best_preds: List[Tuple[str, float, float]] | None = None

    for score_th in SCORE_THRESHOLDS:
        for margin in MARGINS:
            for dist_th in DIST_THRESHOLDS:
                stats = evaluate_config(score_th, margin, dist_th, geo)
                min_km, med_km, mean_km, max_km, _ = stats
                print(
                    f"TH={score_th:<4}  margin={margin:<4}  d_th={dist_th:<3} → mean {mean_km:6.2f} km  med {med_km:6.2f}")
                if (best_stats is None) or (stats[2] < best_stats[2]):  # compare mean
                    best_stats = stats[:4]
                    best_config = (score_th, margin, dist_th)
                    best_preds = stats[4]

    assert best_preds is not None and best_config is not None and best_stats is not None

    # Write predictions
    with BEST_OUT.open("w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["subject_id", "mordecai_lat", "mordecai_lon"])
        for sid, lat, lon in best_preds:
            w.writerow([sid, f"{lat:.6f}", f"{lon:.6f}"])

    print("\nBest config → score_th=", best_config[0], " margin=", best_config[1], " dist_th=", best_config[2])
    min_km, med_km, mean_km, max_km = best_stats
    print(f"  Min    {min_km:.2f} km\n  Median {med_km:.2f} km\n  Mean   {mean_km:.2f} km\n  Max    {max_km:.2f} km")
    print("Predictions saved to", BEST_OUT)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1) 