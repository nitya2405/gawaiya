import json
from pathlib import Path

# Always resolve paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

HINDUSTANI_ROOT = PROJECT_ROOT / "saraga" / "dataset" / "hindustani"
OUTPUT_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_metadata():
    count = 0

    # Level 1: Album folders
    for album_dir in HINDUSTANI_ROOT.iterdir():
        if not album_dir.is_dir():
            continue

        # Level 2: Raag folders inside album
        for raag_dir in album_dir.iterdir():
            if not raag_dir.is_dir():
                continue

            # Each raag folder should contain exactly one JSON
            json_files = list(raag_dir.glob("*.json"))
            if not json_files:
                continue

            json_path = json_files[0]

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                mbid = data.get("mbid")
                if not mbid:
                    continue

                # Raga
                raga = "unknown"
                if data.get("raags"):
                    raga = data["raags"][0].get("common_name", "unknown")

                # Instruments
                instruments = []
                lead_instrument = None
                for artist in data.get("artists", []):
                    inst = artist.get("instrument", {}).get("name")
                    if inst:
                        instruments.append(inst.lower())
                    if artist.get("lead") and inst:
                        lead_instrument = inst.lower()

                unified = {
                    "mbid": mbid,
                    "raga": raga,
                    "taal": data["taals"][0]["name"] if data.get("taals") else "unknown",
                    "laya": data["layas"][0]["name"] if data.get("layas") else "unknown",
                    "instruments": sorted(set(instruments)),
                    "lead_instrument": lead_instrument,
                    "duration_ms": data.get("length", 0),
                    "album": album_dir.name,
                    "raag_folder": raag_dir.name
                }

                out_file = OUTPUT_DIR / f"{mbid}.json"
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(unified, f, indent=2)

                count += 1

            except Exception as e:
                continue

    print(f"Extracted {count} metadata files")

if __name__ == "__main__":
    extract_metadata()
