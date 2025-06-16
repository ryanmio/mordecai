#!/usr/bin/env python3
"""Evaluate Mordecai predictions against ground truth coordinates."""

import csv
import sys
from pathlib import Path
from geographiclib.geodesic import Geodesic

ROOT = Path(__file__).resolve().parent
GROUND_TRUTH = ROOT / "validation - TEST-FULL-H1-final.csv"

def main() -> None:
    # Check command line arguments
    if len(sys.argv) > 1:
        predictions_file = Path(sys.argv[1])
    else:
        # Default to enhanced predictions
        predictions_file = ROOT / "mordecai3_predictions_enhanced.csv"
    
    if not predictions_file.exists():
        sys.exit(f"Predictions file not found: {predictions_file}")
    
    if not GROUND_TRUTH.exists():
        sys.exit(f"Ground truth file not found: {GROUND_TRUTH}")

    # Load ground truth
    gt_coords = {}
    with GROUND_TRUTH.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("has_ground_truth") == "1":
                sid = row["subject_id"]
                try:
                    # Parse latitude/longitude column like "36.934201,-77.389837"
                    coord_str = row.get("latitude/longitude", "").strip()
                    if coord_str:
                        lat_str, lon_str = coord_str.split(",")
                        lat = float(lat_str.strip())
                        lon = float(lon_str.strip())
                        gt_coords[sid] = (lat, lon)
                except (ValueError, KeyError, AttributeError):
                    continue

    # Load predictions
    pred_coords = {}
    with predictions_file.open(newline="") as f:
        for row in csv.DictReader(f):
            sid = row["subject_id"]
            try:
                lat = float(row["mordecai_lat"])
                lon = float(row["mordecai_lon"])
                pred_coords[sid] = (lat, lon)
            except (ValueError, KeyError):
                continue

    # Calculate distances
    geod = Geodesic.WGS84
    distances = []
    missing_predictions = 0

    for sid, (gt_lat, gt_lon) in gt_coords.items():
        if sid in pred_coords:
            pred_lat, pred_lon = pred_coords[sid]
            result = geod.Inverse(gt_lat, gt_lon, pred_lat, pred_lon)
            distance_km = result["s12"] / 1000.0
            distances.append(distance_km)
        else:
            missing_predictions += 1

    if not distances:
        print("No valid predictions found.")
        return

    # Calculate statistics
    distances.sort()
    n = len(distances)
    min_dist = distances[0]
    max_dist = distances[-1]
    mean_dist = sum(distances) / n
    median_dist = distances[n // 2] if n % 2 == 1 else (distances[n // 2 - 1] + distances[n // 2]) / 2

    print(f"Evaluating: {predictions_file.name}")
    print(f"Ground truth entries: {len(gt_coords)}")
    print(f"Valid predictions: {len(distances)}")
    print(f"Missing predictions: {missing_predictions}")
    print()
    print(f"Distance statistics (km):")
    print(f"  Min:    {min_dist:.2f}")
    print(f"  Median: {median_dist:.2f}")
    print(f"  Mean:   {mean_dist:.2f}")
    print(f"  Max:    {max_dist:.2f}")

if __name__ == "__main__":
    main() 