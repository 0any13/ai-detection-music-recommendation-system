import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
FEATURES_FILE = Path("data/tracks_with_features.json")
CACHE_FILE = Path("data/lastfm_tags_cache.json")
REQUEST_DELAY = 0.2
SAVE_EVERY = 200


def fetch_artist_tags(artist: str) -> list:
    try:
        response = requests.get(
            "https://ws.audioscrobbler.com/2.0/",
            params={
                "method": "artist.getTopTags",
                "artist": artist,
                "api_key": LASTFM_API_KEY,
                "format": "json",
                "autocorrect": 1,
            },
            timeout=10,
        )
        data = response.json()
        if "toptags" not in data:
            return []
        #artist tags have 'count' going up to 100 (% of users who applied it)
        #keep tags where at least 5% of tagging users applied it
        return [
            t["name"].lower().strip()
            for t in data["toptags"].get("tag", [])
            if isinstance(t, dict) and int(t.get("count", 0)) >= 5
        ]
    except Exception:
        return []


def main():
    if not LASTFM_API_KEY:
        print("Error: LASTFM_API_KEY not in .env")
        return

    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        features = json.load(f)

    print(f"Tracks with audio features : {len(features):,}")
    print(f"Estimated time             : ~{len(features) * REQUEST_DELAY / 60:.0f} minutes")

    #load existing cache
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Already cached             : {len(cache):,}")
    else:
        cache = {}

    to_process = [t for t in features if t["track_id"] not in cache]
    print(f"Remaining                  : {len(to_process):,}\n")

    if not to_process:
        print("All done.")
        _summary(cache)
        return

    #cache artist tags so we dont re-fetch the same artist repeatedly
    artist_tag_cache = {}
    found = not_found = 0

    for i, track in enumerate(to_process, 1):
        tid = track["track_id"]
        artist = track.get("artist", "").strip()

        if not artist:
            cache[tid] = []
            not_found += 1
        else:
            #reuse cached result if we already fetched this artist
            if artist not in artist_tag_cache:
                artist_tag_cache[artist] = fetch_artist_tags(artist)
                time.sleep(REQUEST_DELAY)

            tags = artist_tag_cache[artist]
            cache[tid] = tags
            if tags:
                found += 1
            else:
                not_found += 1

        if i % 100 == 0:
            pct = i / len(to_process) * 100
            print(f"  {i:,}/{len(to_process):,} ({pct:.1f}%)  "
                  f"found={found}  not_found={not_found}  "
                  f"unique artists queried={len(artist_tag_cache)}")

        if i % SAVE_EVERY == 0:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            print(f"  -- saved ({len(cache):,} total) --")

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    print(f"\nDone. {CACHE_FILE}")
    print(f"  With tags : {found}")
    print(f"  No tags   : {not_found}")
    _summary(cache)


def _summary(cache: dict):
    with_tags = {k: v for k, v in cache.items() if v}
    print(f"\nTracks with tags: {len(with_tags):,} / {len(cache):,}")
    counts = {}
    for tags in with_tags.values():
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
    print("Top 20 tags:")
    for tag, n in sorted(counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {tag:30s} {n}")


if __name__ == "__main__":
    main()