#!/usr/bin/env python3
"""DEV tuning script – iterate on county-centroid fallback heuristics.

Keeps production runner untouched. Writes dev_tuned_predictions.csv and prints
error stats vs gold set.
"""
import csv
import math
import re
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from elasticsearch import Elasticsearch
from mordecai3 import Geoparser

ROOT = Path(__file__).resolve().parent
GT_FILE = ROOT / "validation - TEST-FULL-H1-final.csv"
OUT_FILE = ROOT / "dev_tuned_predictions.csv"

# Helper: Haversine distance in km
R_KM = 6371.0

def haversine(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R_KM * math.atan2(math.sqrt(a), math.sqrt(1-a))

# Bounding box for Virginia (slightly widened)
VA_BOX = dict(min_lat=35.5, max_lat=40.5, min_lon=-85.0, max_lon=-74.5)
STATE_CENTROID = (37.43157, -78.6569)

# County abbreviation expansion map (minimal)
TOKEN_MAP = {
    "Pr": "Prince",
    "Geo": "George",
    "Chas": "Charles",
    "Jas": "James",
    "Spotsyl": "Spotsylvania",
    "Spotsyl.": "Spotsylvania",
    "K": "King",
    "Wm": "William",
    "&": "and",
}

county_regex = re.compile(r"([A-Za-z .&]+?)\s+(County|Co\b|Co[.;,]|City\b|City\s+of)", re.IGNORECASE)


def norm_county(raw: str) -> str:
    """Expand common abbreviations and cleanup punctuation for county look-up."""
    txt = raw.replace(".", " ").replace(",", " ")
    parts = txt.split()
    norm_parts = [TOKEN_MAP.get(p.rstrip('.').rstrip(','), p) for p in parts]
    clean = " ".join(norm_parts)
    return clean.title()


def county_centroid(es: Elasticsearch, county_name: str) -> Optional[Tuple[float, float]]:
    name = norm_county(county_name)
    q = {
        "size": 1,
        "query": {
            "bool": {
                "must": [{"match": {"name": name}}],
                "filter": [
                    {"term": {"admin1_code": "VA"}},
                    {"terms": {"feature_code": ["ADM2", "PPLA2", "PPLA", "PPLA3"]}},
                ],
            }
        },
    }
    try:
        resp = es.search(index="geonames", body=q)
        hits = resp.get("hits", {}).get("hits", [])
        if hits:
            coords = hits[0]["_source"]["coordinates"].split(",")
            return float(coords[0]), float(coords[1])
    except Exception:
        pass
    return None


def in_va(ent: Dict[str, Any]) -> bool:
    try:
        lat = float(ent.get("lat", ent.get("latitude")))
        lon = float(ent.get("lon", ent.get("longitude")))
    except (TypeError, ValueError):
        return False
    return VA_BOX["min_lat"] <= lat <= VA_BOX["max_lat"] and VA_BOX["min_lon"] <= lon <= VA_BOX["max_lon"]


def main() -> None:
    geo = Geoparser()
    es = Elasticsearch(hosts=["http://localhost:9200"], request_timeout=30)

    with OUT_FILE.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["subject_id", "lat", "lon"])
        writer.writeheader()

        dists = []
        with GT_FILE.open() as fgt:
            gt_reader = list(csv.DictReader(fgt))

        for row in gt_reader:
            if row["has_ground_truth"] != "1":
                continue
            sid = row["subject_id"]
            text = row["raw_entry"]

            res = geo.geoparse_doc(text, known_country="USA")
            ents = res.get("geolocated_ents", []) if isinstance(res, dict) else res

            # Heuristic: pick best VA result; if none, fallback to county centroid
            candidates = [e for e in ents if in_va(e)]
            chosen = None
            # ::TUNE:: Require model score >= threshold
            SCORE_TH = 0.15
            if candidates:
                # score field may be missing on some; default 0
                candidates.sort(key=lambda e: -float(e.get("score", 0)))
                if float(candidates[0].get("score", 0)) >= SCORE_TH:
                    chosen = candidates[0]

            lat = lon = None
            if chosen:
                lat = float(chosen.get("lat", chosen.get("latitude")))
                lon = float(chosen.get("lon", chosen.get("longitude")))
            else:
                m = county_regex.search(text)
                if m:
                    centroid = county_centroid(es, m.group(1))
                    if centroid:
                        lat, lon = centroid
            if lat is None:
                lat, lon = STATE_CENTROID

            writer.writerow({"subject_id": sid, "lat": f"{lat:.6f}", "lon": f"{lon:.6f}"})

            # gt distance
            gt_lat, gt_lon = map(float, row["latitude/longitude"].split(","))
            dists.append(haversine(gt_lat, gt_lon, lat, lon))

        print("DEV tuning results (SCORE_TH=", SCORE_TH, ")")
        print(f"   Min   {min(dists):.2f} km")
        print(f"   Median {statistics.median(dists):.2f} km")
        print(f"   Mean  {statistics.mean(dists):.2f} km")
        print(f"   Max   {max(dists):.2f} km")

if __name__ == "__main__":
    main() 