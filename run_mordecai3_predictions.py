import csv
import sys
import re
from pathlib import Path

from typing import List, Dict, Any, Optional, Tuple

from elasticsearch import Elasticsearch

try:
    from mordecai3 import Geoparser  # type: ignore
except ImportError:
    sys.exit("Error: mordecai3 is not installed in this environment. Activate .venv_m3 and `pip install mordecai3`. ")


def _preprocess_text(text: str) -> str:
    """Expand common 18-century abbreviations so spaCy can tag places.

    The mapping is intentionally tiny – just enough to coax spaCy into
    emitting GPE/LOC entities without altering the substantive wording.
    """
    substitutions = {
        r"\bCo[.;,]?\b": "County",
        r"\bRiv[.;,]?\b": "River",
        r"\bCr[.;,]?\b": "Creek",
        r"\bSw[.;,]?\b": "Swamp",
    }
    out = text
    for pat, repl in substitutions.items():
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    # collapse multiple spaces and trim
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip()


def main():
    """Run Mordecai v3 over the ground-truth subset and emit predictions CSV."""
    root = Path(__file__).resolve().parent
    input_csv = root / "validation - TEST-FULL-H1-final.csv"
    if not input_csv.exists():
        sys.exit(f"Input CSV not found: {input_csv}")

    # Instantiate Mordecai 3 once
    print("Loading Mordecai 3 (spaCy transformer) … this may take 30-60 s the first time.")
    try:
        geo = Geoparser()
    except Exception as exc:
        print("FATAL: Could not initialise Mordecai 3 –", exc)
        sys.exit(1)

    # Elasticsearch connection (GeoNames index already running)
    es: Optional[Elasticsearch] = None
    try:
        es = Elasticsearch(hosts=["http://localhost:9200"], request_timeout=30)
    except Exception:
        es = None  # we'll fall back to state centroid only if ES unavailable

    VA_BBOX = {
        "min_lat": 36.0,
        "max_lat": 40.0,
        "min_lon": -84.0,
        "max_lon": -75.0,
    }

    STATE_CENTROID = (37.43157, -78.6569)  # Virginia

    county_regex = re.compile(r"([A-Za-z .]+?)\s+(County|Co\b|Co[.;,]|City\b|City\s+of)", re.IGNORECASE)

    def in_va_bbox(ent: Dict[str, Any]) -> bool:
        try:
            lat = float(ent.get("lat", ent.get("latitude")))
            lon = float(ent.get("lon", ent.get("longitude")))
        except (TypeError, ValueError):
            return False
        if not (VA_BBOX["min_lat"] <= lat <= VA_BBOX["max_lat"]):
            return False
        if not (VA_BBOX["min_lon"] <= lon <= VA_BBOX["max_lon"]):
            return False
        # Country check
        cc = (ent.get("country_code3") or ent.get("country_code") or "").upper()
        return cc in {"USA", "US"}

    def county_centroid(county_raw: str) -> Optional[Tuple[float, float]]:
        """Lookup county centroid in GeoNames ES and return (lat, lon) or None."""
        if not es:
            return None
        county_name = county_raw.strip()
        if not county_name:
            return None
        # Build ES query
        q = {
            "size": 1,
            "query": {
                "bool": {
                    "must": [
                        {"match": {"name": county_name}},
                    ],
                    "filter": [
                        {"term": {"admin1_code": "VA"}},
                        {"terms": {"feature_code": ["ADM2", "PPLA2", "PPLA", "PPLA3"]}},
                    ],
                }
            }
        }
        try:
            resp = es.search(index="geonames", body=q)
            hits = resp.get("hits", {}).get("hits", [])
            if hits:
                src = hits[0]["_source"]
                lat = float(src["coordinates"].split(",")[0])
                lon = float(src["coordinates"].split(",")[1])
                return (lat, lon)
        except Exception:
            pass
        return None

    out_path = root / "mordecai3_predictions.csv"
    with open(out_path, "w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=["subject_id", "mordecai_lat", "mordecai_lon"])
        writer.writeheader()

        total = 0
        success = 0
        va_hits = county_fallbacks = state_fallbacks = 0
        with input_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("has_ground_truth", "0")) != "1":
                    continue
                total += 1
                subj_id = row.get("subject_id")
                raw_text_orig = row.get("raw_entry", "")

                # Pre-process abbreviations to help spaCy recognise GPEs
                text = _preprocess_text(raw_text_orig)

                lat = lon = ""

                try:
                    # Bias to United States to avoid London / Oslo, etc.
                    res = geo.geoparse_doc(text, known_country="USA")

                    ents: List[Dict[str, Any]] = []
                    if isinstance(res, dict):
                        ents = res.get("geolocated_ents", [])
                    elif isinstance(res, list):
                        ents = res

                    # Bounding-box filter (VA only)
                    filtered = [e for e in ents if in_va_bbox(e)]

                    chosen = filtered[0] if filtered else None

                    if chosen:
                        lat_val = chosen.get("lat") or chosen.get("latitude")
                        lon_val = chosen.get("lon") or chosen.get("longitude")
                        if lat_val is not None and lon_val is not None:
                            lat = f"{float(lat_val):.6f}"
                            lon = f"{float(lon_val):.6f}"
                            success += 1
                            va_hits += 1
                    # Fallback: county centroid
                    if not chosen:
                        m = county_regex.search(text)
                        if m:
                            county_name = m.group(1).strip()
                            centroid = county_centroid(county_name)
                            if centroid:
                                lat, lon = f"{centroid[0]:.6f}", f"{centroid[1]:.6f}"
                                county_fallbacks += 1
                            else:
                                lat, lon = f"{STATE_CENTROID[0]:.6f}", f"{STATE_CENTROID[1]:.6f}"
                                state_fallbacks += 1
                        else:
                            # ultimate fallback: state centroid
                            lat, lon = f"{STATE_CENTROID[0]:.6f}", f"{STATE_CENTROID[1]:.6f}"
                            state_fallbacks += 1
                        success += 1  # filled via fallback
                except Exception as e:
                    print(f"   › Mordecai3 failed for subject_id={subj_id}: {e}")
                    # retain empty lat/lon if failure

                writer.writerow({
                    "subject_id": subj_id,
                    "mordecai_lat": lat,
                    "mordecai_lon": lon
                })

    print("\nProcessed", total, "ground-truth rows with Mordecai 3.")
    print("Successful geocodes:", success, "/", total)
    print(f"   • VA hits           : {va_hits}")
    print(f"   • County fallbacks  : {county_fallbacks}")
    print(f"   • State fallbacks   : {state_fallbacks}")
    print("Output written to", out_path)


if __name__ == "__main__":
    main() 