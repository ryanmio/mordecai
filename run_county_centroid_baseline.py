#!/usr/bin/env python3
"""County-Centroid baseline for Virginia land grants.

Extracts county names from raw_entry text, looks up centroids from FIPS data,
and outputs cc_lat, cc_lon coordinates for each ground-truth row.
"""

import csv
import re
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "validation - TEST-FULL-H1-final.csv"
OUTPUT_CSV = ROOT / "county_centroid_predictions.csv"
VA_STATE_CENTROID = (37.4316, -78.6569)


def load_va_county_centroids() -> Dict[str, Tuple[float, float]]:
    """Load Virginia county centroids."""
    # Virginia county centroids from various sources
    return {
        "HANOVER": (37.7448, -77.4464),
        "HENRICO": (37.5131, -77.3465),
        "PRINCE GEORGE": (37.1232, -78.4928),
        "SPOTSYLVANIA": (38.1949, -77.6269),
        "KING WILLIAM": (37.6859, -77.0136),
        "NEW KENT": (37.4975, -76.9990),
        "CHARLES CITY": (37.3571, -77.0747),
        "JAMES CITY": (37.3043, -76.7675),
        "SURRY": (37.1340, -76.8175),
        "ISLE OF WIGHT": (36.9085, -76.6980),
        "NANSEMOND": (36.7845, -76.4210),
        "NORFOLK": (36.8468, -76.2852),
        "PRINCESS ANNE": (36.7335, -76.0436),
        "BRUNSWICK": (36.6915, -77.8121),
        "ESSEX": (37.9141, -76.9326),
        "KING AND QUEEN": (37.6743, -76.8691),
        "GOOCHLAND": (37.7204, -77.8838),
        "CHESTERFIELD": (37.3771, -77.5059),
        "RICHMOND": (37.5407, -77.4360),
        "STAFFORD": (38.4221, -77.4609),
        "WESTMORELAND": (38.1851, -76.7746),
        "NORTHUMBERLAND": (37.9090, -76.3522),
        "LANCASTER": (37.7543, -76.4630),
        "MIDDLESEX": (37.6140, -76.3136),
        "GLOUCESTER": (37.4093, -76.5013),
        "YORK": (37.2379, -76.5094),
        "WARWICK": (37.0871, -76.4730),
        "ELIZABETH CITY": (36.9460, -76.2299),
        "ACCOMACK": (37.7668, -75.6324),
        "NORTHAMPTON": (37.3543, -75.9646),
    }


def extract_county_name(text: str) -> Optional[str]:
    """Extract county name from raw land grant text."""
    # Common patterns for county mentions
    patterns = [
        r"([A-Za-z\s&]+?)\s+County",
        r"([A-Za-z\s&]+?)\s+Co[.,;]?\b",
        r"City\s+of\s+([A-Za-z\s&]+)",
        r"in\s+([A-Za-z\s&]+?)\s+(?:County|Co[.,;]?)",
        r"of\s+([A-Za-z\s&]+?)\s+(?:County|Co[.,;]?)",
        r"([A-Za-z\s&]+?)\s+Co;",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            county = match.strip().upper()
            
            # Handle common abbreviations and variations
            if county.startswith("PR.") or county.startswith("PRI."):
                county = county.replace("PR.", "PRINCE").replace("PRI.", "PRINCE")
            if county == "K. & Q." or county == "K&Q":
                county = "KING AND QUEEN"
            if county == "JAS. CITY" or county == "JAMES C.":
                county = "JAMES CITY"
            if county == "IS. OF WIGHT" or county == "ISLE WIGHT":
                county = "ISLE OF WIGHT"
            if county == "CHAS. CITY" or county == "CHARLES C.":
                county = "CHARLES CITY"
            if county == "SPOTSY" or county == "SPOTSYL":
                county = "SPOTSYLVANIA"
            if county.endswith(" C."):
                county = county[:-3]  # Remove " C." suffix
                
            # Clean up extra spaces
            county = re.sub(r'\s+', ' ', county).strip()
            
            if county and len(county) > 2:  # Avoid single letters
                return county
    
    return None


def main() -> None:
    if not INPUT_CSV.exists():
        sys.exit(f"Input file not found: {INPUT_CSV}")
    
    va_centroids = load_va_county_centroids()
    
    county_hits = 0
    state_fallbacks = 0
    total_processed = 0
    
    with OUTPUT_CSV.open("w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["subject_id", "cc_lat", "cc_lon"])
        writer.writeheader()
        
        with INPUT_CSV.open(newline="") as fin:
            for row in csv.DictReader(fin):
                if row.get("has_ground_truth") != "1":
                    continue
                
                total_processed += 1
                sid = row["subject_id"]
                raw_entry = row["raw_entry"]
                
                # Try to extract county name
                county_name = extract_county_name(raw_entry)
                
                if county_name and county_name in va_centroids:
                    lat, lon = va_centroids[county_name]
                    county_hits += 1
                else:
                    # Fallback to state centroid
                    lat, lon = VA_STATE_CENTROID
                    state_fallbacks += 1
                
                writer.writerow({
                    "subject_id": sid,
                    "cc_lat": f"{lat:.6f}",
                    "cc_lon": f"{lon:.6f}"
                })
    
    print(f"County hits {county_hits} / state fallback {state_fallbacks} (total {total_processed})")
    print(f"Output written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main() 