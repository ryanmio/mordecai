import csv
import sys
from pathlib import Path

try:
    from mordecai import Geoparser
except ImportError:
    sys.exit("Error: Mordecai is not installed. Please install with `pip install mordecai`." )


def main():
    """Run Mordecai v2 over validation CSV and emit predictions CSV.

    The script expects to be executed from the repository root so that the
    validation CSV is located at ./validation - TEST-FULL-H1-final.csv.
    """
    root = Path(__file__).resolve().parent
    input_csv = root / "validation - TEST-FULL-H1-final.csv"
    if not input_csv.exists():
        sys.exit(f"Input CSV not found: {input_csv}")

    # Instantiate Mordecai Geoparser once
    print("Loading Mordecai models… this may take a moment.")
    try:
        geo = Geoparser()
    except Exception as exc:
        print("WARNING: Could not initialise Mordecai (", exc, ") – falling back to blank predictions.")
        geo = None

    # Prepare output
    out_path = root / "mordecai_predictions.csv"
    out_file = open(out_path, "w", newline="")
    writer = csv.DictWriter(out_file, fieldnames=["subject_id", "mordecai_lat", "mordecai_lon"])
    writer.writeheader()

    total = 0
    success = 0

    with input_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("has_ground_truth", "0")) != "1":
                continue  # skip non‐gold rows

            total += 1
            subj_id = row.get("subject_id")
            raw_text = row.get("raw_entry", "")

            lat = lon = ""  # default blanks
            if geo is not None:
                try:
                    res = geo.geoparse(raw_text)
                    if res:
                        first = res[0]
                        lat = f"{float(first['geo']['lat']):.6f}"
                        lon = f"{float(first['geo']['lon']):.6f}"
                        success += 1
                except Exception as e:
                    print(f"   › Mordecai failed for subject_id={subj_id}: {e}")

            writer.writerow({
                "subject_id": subj_id,
                "mordecai_lat": lat,
                "mordecai_lon": lon
            })

    out_file.close()

    print("\nProcessed", total, "ground‐truth rows.")
    print("Successful geocodes:", success, "/", total)
    print("Output written to", out_path)


if __name__ == "__main__":
    main() 