#!/usr/bin/env python3
"""Enhanced Mordecai-3 runner with aggressive preprocessing and minimal fallbacks.

Goal: Get Mordecai predictions for all 45 rows with <10 fallbacks total.

Strategy:
1. Aggressive text preprocessing to help spaCy find place names
2. Multiple Mordecai attempts with different text variations
3. Minimal fallbacks only when Mordecai completely fails
"""

import csv
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from elasticsearch import Elasticsearch

try:
    from mordecai3 import Geoparser
except ImportError:
    sys.exit("Error: mordecai3 is not installed. Activate .venv_m3 and `pip install mordecai3`.")


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "validation - TEST-FULL-H1-final.csv"
OUTFILE = ROOT / "mordecai3_predictions_enhanced.csv"

# Virginia bounding box for filtering
VA_BBOX = {"min_lat": 36.0, "max_lat": 40.0, "min_lon": -84.0, "max_lon": -75.0}
STATE_CENTROID = (37.43157, -78.6569)


def aggressive_preprocess(text: str) -> List[str]:
    """Generate multiple preprocessed versions to maximize spaCy entity detection."""
    variations = []
    
    # Original text
    variations.append(text)
    
    # Basic abbreviation expansion
    expanded = text
    abbrevs = {
        r"\bCo[.;,]?\b": "County",
        r"\bRiv[.;,]?\b": "River", 
        r"\bCr[.;,]?\b": "Creek",
        r"\bSw[.;,]?\b": "Swamp",
        r"\bPar[.;,]?\b": "Parish",
        r"\bAcs?\b": "acres",
        r"\bAdj\.\s*": "adjacent to ",
        r"\bBeg\.\s*": "beginning ",
        r"\bS\.\s*side": "South side",
        r"\bN\.\s*side": "North side",
        r"\bE\.\s*side": "East side", 
        r"\bW\.\s*side": "West side",
        r"\bBr\.\s*": "Branch ",
        r"\bMt\.\s*": "Mount ",
        r"\bSt\.\s*": "Saint ",
    }
    for pattern, replacement in abbrevs.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    expanded = re.sub(r"\s{2,}", " ", expanded).strip()
    variations.append(expanded)
    
    # Extract just the geographical context (first ~100 chars with place info)
    geo_context = re.search(r"([^;]{0,100}(?:County|Co|River|Riv|Creek|Cr|Swamp|Sw|Parish|Par)[^;]{0,50})", 
                           expanded, re.IGNORECASE)
    if geo_context:
        variations.append(geo_context.group(1).strip())
    
    # Clean version - remove parenthetical info, dates, measurements
    clean = re.sub(r"\([^)]*\)", "", expanded)
    clean = re.sub(r"\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", "", clean)
    clean = re.sub(r"\d+\s+acs?\.?", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r";\s*\d+.*$", "", clean)  # Remove trailing measurements/dates
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    if clean and clean != expanded:
        variations.append(clean)
    
    return variations


def in_virginia_bbox(ent: Dict[str, Any]) -> bool:
    """Check if entity is within Virginia bounding box."""
    try:
        lat = float(ent.get("lat", ent.get("latitude")))
        lon = float(ent.get("lon", ent.get("longitude")))
        if not (VA_BBOX["min_lat"] <= lat <= VA_BBOX["max_lat"]):
            return False
        if not (VA_BBOX["min_lon"] <= lon <= VA_BBOX["max_lon"]):
            return False
        # Check country code
        cc = (ent.get("country_code3") or ent.get("country_code") or "").upper()
        return cc in {"USA", "US"}
    except (TypeError, ValueError):
        return False


def extract_county_fallback(text: str) -> Optional[Tuple[float, float]]:
    """Extract county name and return its centroid if found."""
    # Simple county centroids for common VA counties mentioned in data
    va_counties = {
        "Hanover": (37.7448, -77.4464),
        "Henrico": (37.5131, -77.3465), 
        "Prince George": (37.1232, -78.4928),
        "Spotsylvania": (38.1949, -77.6269),
        "King William": (37.6859, -77.0136),
        "New Kent": (37.4975, -76.9990),
        "Charles City": (37.3571, -77.0747),
        "James City": (37.3043, -76.7675),
        "Surry": (37.1340, -76.8175),
        "Isle Of Wight": (36.9085, -76.6980),
        "Nansemond": (36.7845, -76.4210),
        "Norfolk": (36.8468, -76.2852),
        "Princess Anne": (36.7335, -76.0436),
        "Brunswick": (36.6915, -77.8121),
        "Essex": (37.9141, -76.9326),
        "King And Queen": (37.6743, -76.8691),
        "Goochland": (37.7204, -77.8838),
    }
    
    # Try to extract county name
    patterns = [
        r"([A-Za-z\s&]+?)\s+County",
        r"([A-Za-z\s&]+?)\s+Co[.;,]?\b",
        r"in\s+([A-Za-z\s&]+?)\s+(?:County|Co)",
        r"of\s+([A-Za-z\s&]+?)\s+(?:County|Co)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            county = match.strip().title()
            # Handle common variations
            if county.startswith("Pr."):
                county = county.replace("Pr.", "Prince")
            if county == "K. & Q.":
                county = "King And Queen"
            if county == "Jas. City":
                county = "James City"
            if county == "Is. Of Wight":
                county = "Isle Of Wight"
            
            if county in va_counties:
                return va_counties[county]
    
    return None


def main() -> None:
    if not INPUT.exists():
        sys.exit(f"Ground-truth CSV not found: {INPUT}")

    print("Loading Mordecai 3...")
    geo = Geoparser()
    
    # ES connection
    _ = Elasticsearch(hosts=["http://localhost:9200"], request_timeout=30)

    with OUTFILE.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["subject_id", "mordecai_lat", "mordecai_lon"])
        writer.writeheader()

        total = mordecai_hits = county_fallbacks = state_fallbacks = 0
        
        with INPUT.open(newline="") as fin:
            for row in csv.DictReader(fin):
                if row.get("has_ground_truth") != "1":
                    continue
                    
                total += 1
                sid = row["subject_id"]
                original_text = row["raw_entry"]
                
                lat = lon = ""
                found_coordinate = False
                
                # Try multiple text variations with Mordecai
                text_variations = aggressive_preprocess(original_text)
                
                for i, text_var in enumerate(text_variations):
                    if found_coordinate:
                        break
                        
                    try:
                        res = geo.geoparse_doc(text_var, known_country="USA")
                        ents: List[Dict[str, Any]] = res.get("geolocated_ents", []) if isinstance(res, dict) else res
                        
                        # Filter to Virginia entities and sort by confidence
                        va_ents = [e for e in ents if in_virginia_bbox(e)]
                        if va_ents:
                            # Sort by score/confidence if available
                            va_ents.sort(key=lambda x: x.get("score", x.get("confidence", 0.0)), reverse=True)
                            ent = va_ents[0]
                            lat = f"{float(ent.get('lat', ent.get('latitude'))):.6f}"
                            lon = f"{float(ent.get('lon', ent.get('longitude'))):.6f}"
                            mordecai_hits += 1
                            found_coordinate = True
                            break
                            
                    except Exception as e:
                        continue  # Try next variation
                
                # If no Mordecai hit, try county fallback
                if not found_coordinate:
                    county_coord = extract_county_fallback(original_text)
                    if county_coord:
                        lat, lon = f"{county_coord[0]:.6f}", f"{county_coord[1]:.6f}"
                        county_fallbacks += 1
                        found_coordinate = True
                
                # Last resort: state centroid
                if not found_coordinate:
                    lat, lon = f"{STATE_CENTROID[0]:.6f}", f"{STATE_CENTROID[1]:.6f}"
                    state_fallbacks += 1
                
                writer.writerow({"subject_id": sid, "mordecai_lat": lat, "mordecai_lon": lon})
                
                if total % 10 == 0:
                    print(f"Processed {total}/45 rows...")

        print(f"\nProcessed {total} ground-truth rows.")
        print(f"Mordecai hits: {mordecai_hits} / {total}")
        print(f"County fallbacks: {county_fallbacks}")
        print(f"State fallbacks: {state_fallbacks}")
        print(f"Total fallbacks: {county_fallbacks + state_fallbacks}")
        print("Output →", OUTFILE)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1) 