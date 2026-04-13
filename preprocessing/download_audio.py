import json
import os
from pathlib import Path
from compmusic import dunya

# =========================
# CONFIG
# =========================

API_TOKEN = os.environ.get("DUNYA_TOKEN")
if not API_TOKEN:
    raise RuntimeError("Missing DUNYA_TOKEN env var. Set it to your Dunya API token before running.")

# =========================
# PATH SETUP
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "data" / "saraga_mp3"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# AUTH
# =========================

dunya.set_token(API_TOKEN)

# =========================
# HELPERS
# =========================

def find_collection(mbid):
    """
    Find which Dunya collection contains this MBID
    """
    collections = dunya.get_collections()

    for col in collections:
        slug = col["slug"]

        try:
            files = dunya.get_recording_files(
                mbid,
                collection=slug
            )

            if files:
                return slug

        except:
            continue

    return None


# =========================
# MAIN
# =========================

def download_all():

    meta_files = list(METADATA_DIR.glob("*.json"))

    print(f"Found {len(meta_files)} recordings")

    success = 0
    failed = 0

    for meta_file in meta_files:

        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        mbid = data["mbid"]

        try:
            print(f"\nChecking {mbid}...")

            collection = find_collection(mbid)

            if not collection:
                print("  No collection found")
                failed += 1
                continue

            print(f"  Found in: {collection}")

            dunya.download_recording(
                mbid,
                output_dir=str(OUTPUT_DIR),
                files=["mp3"],
                collection=collection
            )

            print("  Downloaded")

            success += 1

        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1


    print("\n==== DONE ====")
    print(f"Success: {success}")
    print(f"Failed : {failed}")


if __name__ == "__main__":
    download_all()
